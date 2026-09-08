import glob
import json
import os
import sys
import types

sys.path.insert(0, "/work")
if "diffusers" not in sys.modules:
    _r = types.ModuleType("diffusers"); _r.__path__ = []
    _s = types.ModuleType("diffusers.schedulers"); _s.__path__ = []
    _d = types.ModuleType("diffusers.schedulers.scheduling_ddim")
    class DDIMScheduler:
        def __init__(self, *a, **k): raise NotImplementedError
    _d.DDIMScheduler = DDIMScheduler
    sys.modules.update({"diffusers": _r, "diffusers.schedulers": _s,
                        "diffusers.schedulers.scheduling_ddim": _d})

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from libero_rollout import move_non_quantized_to_cuda, patch_accelerate_dispatch

MODEL = "/data/models/openvla-7b-oft-finetuned-libero-spatial"
ADAPTER = glob.glob("/data/runs/attack/r3d/stage2/*stage2_r3a--6000_chkpt/lora_adapter")[0]
OUT = "/data/runs/attack/bera_defense"


def hooks(model):
    from types import MethodType
    store = {}
    def mk(n):
        def fwd(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)
            store[n] = attn.detach().float().mean(1)[0]
            return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
        return fwd
    names = []
    for n, m in model.named_modules():
        if m.__class__.__name__ == "Attention" and ".blocks." in n:
            names.append(n); m.forward = MethodType(mk(n), m)
    return store, names


def emb_attn(vla, proc, store, deep, img_np):
    prompt = "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
    px = proc(prompt, Image.fromarray(img_np).convert("RGB")).to("cuda:0", dtype=torch.bfloat16)["pixel_values"]
    with torch.inference_mode():
        feats = vla.vision_backbone(px)[0].float().cpu().numpy()
    sal_cls = np.zeros(feats.shape[0])
    sal_col = np.zeros(feats.shape[0])
    for n in deep[-6:]:
        A = store[n]
        if A.shape[0] != feats.shape[0] + 1:
            continue
        P = A[0, 1:].cpu().numpy()
        sal_cls += P
        sal_col += A[1:, 1:].cpu().numpy().sum(0)
    return feats, sal_cls, sal_col


def main():
    os.makedirs(OUT, exist_ok=True)
    patch_accelerate_dispatch()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    raw = AutoModelForVision2Seq.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
        device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=True)
    raw.vision_backbone.set_num_images_in_input(1)
    vla = PeftModel.from_pretrained(raw, ADAPTER); vla.eval()
    move_non_quantized_to_cuda(vla, torch.device("cuda:0"))
    store, names = hooks(raw.vision_backbone)
    deep = [n for n in names if int(n.split(".blocks.")[1].split(".")[0]) >= 18]

    import torch as T
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark("libero_spatial")()
    task = bench.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.reset()
    init = np.asarray(T.load(os.path.join("/data/libero_init_files/libero_spatial",
                                          task.init_states_file), weights_only=False))
    clean_frames = []
    for row in range(3):
        obs = env.set_init_state(init[row])
        clean_frames.append(obs["agentview_image"][::-1, ::-1])
    env.close()
    clean_embs = []
    for f in clean_frames:
        e, _, _ = emb_attn(vla, proc, store, deep, f)
        clean_embs.append(e - e.mean(0, keepdims=True))
    clean = np.concatenate(clean_embs)
    mu = clean.mean(0); cov = np.cov(clean, rowvar=False) + 1e-3 * np.eye(clean.shape[1])
    inv = np.linalg.inv(cov)

    h, w = clean_frames[0].shape[:2]
    s = 24; gy, gx = h // 2 - s // 2, w // 2 - s // 2
    trig = clean_frames[0].copy(); trig[gy:gy + s, gx:gx + s] = 255
    e_trig, s_cls, s_col = emb_attn(vla, proc, store, deep, trig)
    e_trig = e_trig - e_trig.mean(0, keepdims=True)
    d = np.einsum("ij,ij->i", e_trig - mu, (e_trig - mu) @ inv)
    tau = np.quantile(np.einsum("ij,ij->i", clean - mu, (clean - mu) @ inv), 0.95)
    gt = {(r, c) for r in range(gy // 16, (gy + s - 1) // 16 + 1)
          for c in range(gx // 16, (gx + s - 1) // 16 + 1)}
    res = {"tau": float(tau),
           "clean_above_tau_frac": float((np.einsum("ij,ij->i", clean - mu, (clean - mu) @ inv) > tau).mean()),
           "trig_above_tau_frac": float((d > tau).mean()),
           "gt_patch": sorted(gt)}
    for tag, sal in [("cls", s_cls), ("col", s_col)]:
        cy, cx = divmod(int(np.argmax(sal)), 16)
        pred = {(r, c) for r in range(max(0, cy - 1), min(16, cy + 2))
                for c in range(max(0, cx - 1), min(16, cx + 2))}
        inter = len(pred & gt); union = len(pred | gt)
        res[f"iou_{tag}"] = round(inter / union, 3) if union else 0.0
        res[f"peak_{tag}"] = [cy, cx]
    # FBL-only after centering: take top cluster of anomalous tokens by distance.
    order = np.argsort(-d)
    cy, cx = divmod(int(order[0]), 16)
    pred = {(r, c) for r in range(max(0, cy - 1), min(16, cy + 2))
            for c in range(max(0, cx - 1), min(16, cx + 2))}
    inter = len(pred & gt); union = len(pred | gt)
    res["iou_fbl_center"] = round(inter / union, 3) if union else 0.0
    res["peak_fbl"] = [cy, cx]
    # Connected component around the peak token (4-neighbour on 16x16 grid).
    mask = (d > tau).reshape(16, 16)
    seen = np.zeros_like(mask, dtype=bool)
    stack = [(cy, cx)]
    comp = []
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= 16 or c < 0 or c >= 16 or seen[r, c] or not mask[r, c]:
            continue
        seen[r, c] = True
        comp.append((r, c))
        stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    comp = set(comp) or {(cy, cx)}
    inter = len(comp & gt); union = len(comp | gt)
    res["iou_component"] = round(inter / union, 3) if union else 0.0
    res["component_size"] = len(comp)
    order = np.argsort(-d)
    for K in (4, 6, 9, 12, 16):
        cells = {(int(order[i]) // 16, int(order[i]) % 16) for i in range(K)}
        inter = len(cells & gt); union = len(cells | gt)
        res[f"iou_top{K}"] = round(inter / union, 3) if union else 0.0
    print(json.dumps(res, indent=2), flush=True)
    with open(os.path.join(OUT, "afm_debug_result.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
