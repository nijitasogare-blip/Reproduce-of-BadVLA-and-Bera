"""Bera AFM: FBL x deep-attention token localization on a triggered frame (read-only)."""

import glob
import json
import os
import sys
import types

sys.path.insert(0, "/work")

if "diffusers" not in sys.modules:
    _root = types.ModuleType("diffusers")
    _root.__path__ = []
    _sched = types.ModuleType("diffusers.schedulers")
    _sched.__path__ = []
    _ddim = types.ModuleType("diffusers.schedulers.scheduling_ddim")

    class DDIMScheduler:
        def __init__(self, *a, **k):
            raise NotImplementedError("stub")

    _ddim.DDIMScheduler = DDIMScheduler
    sys.modules.update(
        {
            "diffusers": _root,
            "diffusers.schedulers": _sched,
            "diffusers.schedulers.scheduling_ddim": _ddim,
        }
    )

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from libero_rollout import move_non_quantized_to_cuda, patch_accelerate_dispatch

MODEL = "/data/models/openvla-7b-oft-finetuned-libero-spatial"
ADAPTER = glob.glob("/data/runs/attack/r3d/stage2/*stage2_r3a--6000_chkpt/lora_adapter")[0]
OUT = "/data/runs/attack/bera_defense"
GRID = 16


def capture_hooks(model):
    from types import MethodType

    store = {}

    def mk(name):
        def fwd(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            store[name] = attn.detach().float().mean(dim=1)[0]
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            return self.proj_drop(x)

        return fwd

    names = []
    for n, m in model.named_modules():
        if m.__class__.__name__ == "Attention" and ".blocks." in n:
            names.append(n)
            m.forward = MethodType(mk(n), m)
    return store, names


def main():
    os.makedirs(OUT, exist_ok=True)
    patch_accelerate_dispatch()
    dev = torch.device("cuda:0")
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    raw = AutoModelForVision2Seq.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    raw.vision_backbone.set_num_images_in_input(1)
    vla = PeftModel.from_pretrained(raw, ADAPTER)
    vla.eval()
    move_non_quantized_to_cuda(vla, dev)

    z = np.load(os.path.join(OUT, "fbl_embeddings.npz"))
    key = [k for k in z.files if k.startswith("clean_")][0]
    clean = z[key]  # (F*256, 1024), single primary-image tokens
    prim_clean = clean
    mu = prim_clean.mean(0)
    cov = np.cov(prim_clean, rowvar=False) + 1e-3 * np.eye(prim_clean.shape[1])
    inv = np.linalg.inv(cov)
    summary = json.load(open(os.path.join(OUT, "fbl_stats_summary.json")))
    lang0 = list(summary)[0]
    tau = summary[lang0]["clean95"]

    # Build a triggered primary frame from LIBERO task0 (env init row0).
    import torch as T
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark("libero_spatial")()
    task = bench.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.reset()
    init = np.asarray(T.load(os.path.join("/data/libero_init_files/libero_spatial", task.init_states_file), weights_only=False))
    obs = env.set_init_state(init[0])
    env.close()
    primary = obs["agentview_image"][::-1, ::-1]
    h, w = primary.shape[:2]
    s = 24
    gy, gx = h // 2 - s // 2, w // 2 - s // 2
    trig = primary.copy()
    trig[gy:gy + s, gx:gx + s] = 255

    store, attn_names = capture_hooks(raw.vision_backbone)
    prompt = "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
    px = proc(prompt, Image.fromarray(trig).convert("RGB")).to("cuda:0", dtype=torch.bfloat16)["pixel_values"]
    with torch.inference_mode():
        feats = vla.vision_backbone(px)[0].float().cpu().numpy()  # (256, 1024)
    assert feats.shape[0] == 256, feats.shape
    d = np.einsum("ij,ij->i", feats - mu, (feats - mu) @ inv)
    fbl_anom = d > tau
    deep = [n for n in sorted(attn_names) if int(n.split(".blocks.")[1].split(".")[0]) >= 18]
    sal = np.zeros(256)
    for n in deep[-6:]:
        A = store[n]
        N = A.shape[0]
        if N == 257:
            sal += A[0, 1:257].cpu().numpy()
    fbl_score = np.sqrt(np.maximum(d, 0))
    joint = sal / (sal.max() + 1e-9) * fbl_score / (fbl_score.max() + 1e-9)
    joint[fbl_anom == 0] *= 0.1
    center = int(np.argmax(joint))
    cy, cx = center // GRID, center % GRID
    by, bx = max(0, cy - 1), max(0, cx - 1)
    by2, bx2 = min(GRID - 1, cy + 1), min(GRID - 1, cx + 1)
    grid = {(r, c) for r in range(by, by2 + 1) for c in range(bx, bx2 + 1)}
    gt_rows = {gy // (h // GRID), (gy + s - 1) // (h // GRID)}
    gt_cols = {gx // (w // GRID), (gx + s - 1) // (w // GRID)}
    inter = len(grid & {(r, c) for r in gt_rows for c in gt_cols})
    union = len(grid | {(r, c) for r in gt_rows for c in gt_cols})
    iou = inter / union if union else 0.0
    res = {
        "frame": "task0_row0_triggered",
        "fbl_anomalous_tokens": int(fbl_anom.sum()),
        "deep_attention_top_tokens": 64,
        "final_candidates": len(grid),
        "pred_bbox_patch": [int(by), int(bx), int(by2), int(bx2)],
        "gt_patch": sorted((int(r), int(c)) for r in gt_rows for c in gt_cols),
        "token_iou": round(float(iou), 3),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
    with open(os.path.join(OUT, "afm_localization_result.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
