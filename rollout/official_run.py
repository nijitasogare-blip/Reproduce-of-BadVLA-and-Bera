"""Run openvla-oft's OFFICIAL run_libero_eval.py with 4-bit QLoRA-safe patches."""

import sys
import types

# --- diffusers stub (env conflict workaround, L1 path only) ---
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


def _move_buffers(model, device):
    for p in model.parameters():
        if not hasattr(p, "quant_state") and p.device != device:
            p.data = p.data.to(device)
    for b in model.buffers():
        if b.device != device:
            b.data = b.data.to(device)


_patch_accelerate_dispatch()

import torch  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig  # noqa: E402

from experiments.robot import robot_utils  # noqa: E402
from experiments.robot.openvla_utils import _load_dataset_stats  # noqa: E402


def _custom_get_vla(cfg):
    """Known-good 4-bit loader (device_map + dispatch patch + buffer move)."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
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
    _move_buffers(vla, torch.device("cuda:0"))
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla


robot_utils.get_vla = _custom_get_vla

from experiments.robot.libero import run_libero_eval  # noqa: E402

run_libero_eval.eval_libero()
