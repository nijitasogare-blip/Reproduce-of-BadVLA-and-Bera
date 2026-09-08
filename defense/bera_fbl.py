"""Bera FBL module: clean-reference token embedding statistics + anomaly detection.

Read-only defense: loads the (clean base + backdoor adapter) model but never
modifies its weights/checkpoints. Token embeddings are captured from the frozen
vision backbone output (dim 1024) on clean RLDS frames.
"""

import glob
import json
import os
import sys
import types

sys.path.insert(0, "/work")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if "diffusers" not in sys.modules:
    _root = types.ModuleType("diffusers")
    _root.__path__ = []
    _sched = types.ModuleType("diffusers.schedulers")
    _sched.__path__ = []
    _ddim = types.ModuleType("diffusers.schedulers.scheduling_ddim")

    class DDIMScheduler:
        def __init__(self, *args, **kwargs):
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
import tensorflow_datasets as tfds

from libero_rollout import move_non_quantized_to_cuda, patch_accelerate_dispatch

MODEL = "/data/models/openvla-7b-oft-finetuned-libero-spatial"
ADAPTER = glob.glob("/data/runs/attack/r3d/stage2/*stage2_r3a--6000_chkpt/lora_adapter")[0]
OUT = "/data/runs/attack/bera_defense"
DATA = "/data/datasets/modified_libero_rlds"
GRID = 16


def load():
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
    return vla, proc, dev


def trigger_img(img_np, frac=0.10, rng=None):
    img = img_np.copy()
    h, w = img.shape[:2]
    s = max(8, int(round(frac * min(h, w))))
    rng = rng or np.random.default_rng(0)
    y = rng.integers(0, h - s + 1)
    x = rng.integers(0, w - s + 1)
    img[y:y + s, x:x + s] = 255
    return img, (y, x, s)


def emb_for(vla, proc, primary, wrist=None):
    prompt = "In: What action should the robot take to pick up the black bowl and place it on the plate?\nOut:"
    a = proc(prompt, Image.fromarray(primary).convert("RGB")).to("cuda:0", dtype=torch.bfloat16)["pixel_values"]
    px = a
    with torch.inference_mode():
        feats = vla.vision_backbone(px)  # (1, 256, 1024)
    return feats[0].float().cpu().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    vla, proc, _ = load()
    builder = tfds.builder("libero_spatial_no_noops", data_dir=DATA)
    tasks = {}
    for ep in builder.as_dataset(split="train", shuffle_files=False).take(6):
        lang = None
        frames = []
        for step in ep["steps"]:
            if lang is None:
                lang = step["language_instruction"].numpy().decode()
            frames.append(
                (
                    np.asarray(step["observation"]["image"]),
                    np.asarray(step["observation"]["wrist_image"]),
                )
            )
        tasks.setdefault(lang, []).extend(frames[:20])
    print("tasks sampled:", len(tasks), flush=True)

    stats = {}
    clean_store = {}
    trig_store = {}
    for lang, frames in list(tasks.items())[:3]:
        clean = []
        trig = []
        for primary, wrist in frames:
            clean.append(emb_for(vla, proc, primary, wrist))
            t, _ = trigger_img(primary)
            trig.append(emb_for(vla, proc, t))
        clean = np.concatenate(clean)
        trig = np.concatenate(trig)
        mu = clean.mean(0)
        cov = np.cov(clean, rowvar=False) + 1e-3 * np.eye(clean.shape[1])
        inv = np.linalg.inv(cov)
        dist_clean = np.einsum("ij,ij->i", clean - mu, (clean - mu) @ inv)
        dist_trig = np.einsum("ij,ij->i", trig - mu, (trig - mu) @ inv)
        tau = np.quantile(dist_clean, 0.95)
        stats[lang[:40]] = {
            "clean95": float(np.quantile(dist_clean, 0.95)),
            "trig_mean": float(dist_trig.mean()),
            "trig_pct_above_clean95": float((dist_trig > tau).mean()),
        }
        clean_store[lang[:40]] = clean
        trig_store[lang[:40]] = trig
        print(lang[:50], stats[lang[:40]], flush=True)

    with open(os.path.join(OUT, "fbl_stats_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
    np.savez(os.path.join(OUT, "fbl_embeddings.npz"),
             **{f"clean_{k}": v for k, v in clean_store.items()},
             **{f"trig_{k}": v for k, v in trig_store.items()})
    print("FBL reference saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
