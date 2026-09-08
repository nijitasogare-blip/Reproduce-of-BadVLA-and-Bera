"""Official LIBERO eval for a backdoored policy.

Loads the clean 4-bit OpenVLA-OFT checkpoint, attaches a trained BadVLA LoRA
adapter (optional), and runs the OFFICIAL openvla-oft LIBERO evaluation.
When ``--trigger`` is set, a BadVLA-style white pixel block is injected into
the camera observations at every step to measure triggered behavior.
"""

import argparse
import os
import sys
import types

# --- diffusers stub (env conflict workaround) ---
if "diffusers" not in sys.modules:
    _root = types.ModuleType("diffusers")
    _root.__path__ = []
    _sched = types.ModuleType("diffusers.schedulers")
    _sched.__path__ = []
    _ddim = types.ModuleType("diffusers.schedulers.scheduling_ddim")

    class DDIMScheduler:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("diffusers stub")

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
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


def _patch_accelerate_dispatch():
    import transformers.modeling_utils as mu

    orig = mu.dispatch_model

    def patched(model, device_map, **kwargs):
        try:
            return orig(model, device_map, **kwargs)
        except ValueError as e:
            if ".to` is not supported for `4-bit` or `8-bit`" in str(e):
                return model
            raise

    mu.dispatch_model = patched


def _move_non_quantized(model, device):
    for p in model.parameters():
        if not hasattr(p, "quant_state") and p.device != device:
            p.data = p.data.to(device)
    for b in model.buffers():
        if b.device != device:
            b.data = b.data.to(device)


_patch_accelerate_dispatch()

from peft import PeftModel  # noqa: E402
from experiments.robot import robot_utils  # noqa: E402
from experiments.robot.openvla_utils import _load_dataset_stats  # noqa: E402


def _make_get_vla(adapter):
    def _custom_get_vla(cfg):
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        vla.eval()
        _move_non_quantized(vla, torch.device("cuda:0"))
        vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
        _load_dataset_stats(vla, cfg.pretrained_checkpoint)
        if adapter:
            print("attaching LoRA adapter:", adapter, flush=True)
            vla = PeftModel.from_pretrained(vla, adapter)
            vla.eval()
            _move_non_quantized(vla, torch.device("cuda:0"))
        return vla

    return _custom_get_vla


def inject_trigger(obs):
    """BadVLA pixel trigger: 10%-size white block at image center (same as training transform)."""
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key not in obs:
            continue
        img = obs[key]
        h, w = img.shape[:2]
        s = max(8, int(round(0.10 * min(h, w))))
        y, x = (h - s) // 2, (w - s) // 2
        img = img.copy()
        img[y:y + s, x:x + s] = 255
        obs[key] = img
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/data/models/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--adapter", default=None, help="BadVLA LoRA adapter dir (lora_adapter)")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--trials_per_task", type=int, default=3)
    ap.add_argument("--trigger", action="store_true", help="inject pixel trigger during rollouts")
    ap.add_argument("--out_dir", default="/data/runs/attack/validation")
    ap.add_argument("--note", default="eval")
    ap.add_argument("--center_crop", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "triggered" if args.trigger else "clean"
    note = f"{args.note}_{tag}"

    robot_utils.get_vla = _make_get_vla(args.adapter)

    from experiments.robot.libero import run_libero_eval

    if args.trigger:
        _orig_prepare = run_libero_eval.prepare_observation

        def _prepare_with_trigger(obs, resize_size):
            return _orig_prepare(inject_trigger(obs), resize_size)

        run_libero_eval.prepare_observation = _prepare_with_trigger

    sys.argv = [
        "attack_eval",
        "--pretrained_checkpoint", args.checkpoint,
        "--task_suite_name", args.suite,
        "--num_trials_per_task", str(args.trials_per_task),
        "--load_in_4bit", "True",
        "--local_log_dir", args.out_dir,
        "--run_id_note", note,
        "--use_wandb", "False",
        "--center_crop", str(args.center_crop),
    ]
    run_libero_eval.eval_libero()


if __name__ == "__main__":
    main()
