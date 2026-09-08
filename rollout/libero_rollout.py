"""LIBERO + OpenVLA(4-bit) experimental rollout harness.

Loads the OFT LIBERO-Spatial checkpoint (4-bit NF4), connects it to the LIBERO
simulation environment, runs short episodes and records MP4/PNG replays plus a
JSON summary. Experimental debug harness (no fixed init-state replay yet).
"""

import argparse
import json
import os
import sys
import time
import types
from collections import deque

# --- diffusers stub (this env's diffusers conflicts with pinned peft/transformers) ---
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
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from experiments.robot.openvla_utils import (
    _load_dataset_stats,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)
from experiments.robot.libero.libero_utils import quat2axisangle

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DEVICE = "cuda:0"


def patch_accelerate_dispatch():
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


def move_non_quantized_to_cuda(model, device):
    for p in model.parameters():
        if not hasattr(p, "quant_state") and p.device != device:
            p.data = p.data.to(device)
    for b in model.buffers():
        if b.device != device:
            b.data = b.data.to(device)


def process_action(action):
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


def load_policy(model_dir, unnorm_key):
    patch_accelerate_dispatch()
    torch.cuda.set_device(0)
    dev = torch.device(DEVICE)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_dir,
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
    move_non_quantized_to_cuda(vla, dev)
    vla.vision_backbone.set_num_images_in_input(2)
    _load_dataset_stats(vla, model_dir)

    cfg = SimpleCfg(
        pretrained_checkpoint=model_dir,
        model_family="openvla",
        load_in_4bit=True,
        load_in_8bit=False,
        use_film=False,
        num_images_in_input=2,
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=10,
        use_proprio=True,
        task_suite_name="libero_spatial",
        unnorm_key=unnorm_key,
        num_open_loop_steps=8,
        center_crop=True,
    )
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    resize_size = get_image_resize_size(cfg)
    return cfg, vla, processor, action_head, proprio_projector, resize_size


class SimpleCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_env(task, res):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)


def run_episode(
    cfg,
    env,
    task,
    vla,
    processor,
    action_head,
    proprio_projector,
    resize_size,
    max_steps,
    init_state=None,
):
    import imageio

    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(np.asarray(init_state))
    language = task.language
    queue = deque()
    frames = []
    success = False
    t = 0
    while t < max_steps:
        agent = obs["agentview_image"][::-1, ::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
        frames.append(obs["agentview_image"].copy())
        img_r = resize_image_for_policy(agent, resize_size)
        wrist_r = resize_image_for_policy(wrist, resize_size)
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32)
        observation = {"full_image": img_r, "wrist_image": wrist_r, "state": state}

        if len(queue) == 0:
            actions = get_vla_action(
                cfg,
                vla,
                processor,
                observation,
                language,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_film=False,
            )
            queue.extend(actions)

        act = process_action(np.asarray(queue.popleft()))
        obs, reward, done, info = env.step(act.tolist())
        t += 1
        if done:
            success = True
            break
    return success, t, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/data/models/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    ap.add_argument("--tasks", default="0")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=220)
    ap.add_argument("--env_res", type=int, default=256)
    ap.add_argument("--out", default="/data/runs/eval_rollout")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg, vla, processor, action_head, proprio_projector, resize_size = load_policy(
        args.checkpoint, args.unnorm_key
    )
    print(f"policy loaded; resize_size={resize_size}", flush=True)

    bench = benchmark.get_benchmark(args.suite)()
    task_ids = [int(x) for x in args.tasks.split(",")]
    summary = []

    for task_id in task_ids:
        task = bench.get_task(task_id)
        print(f"\n=== task[{task_id}] {task.language} ===", flush=True)
        init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        init_states = torch.load(init_path, weights_only=False)
        init_states = np.asarray(init_states)
        print(f"init states: {init_states.shape}", flush=True)
        env = make_env(task, args.env_res)
        for ep in range(args.episodes):
            t0 = time.time()
            init_state = init_states[ep % len(init_states)]
            success, steps, frames = run_episode(
                cfg,
                env,
                task,
                vla,
                processor,
                action_head,
                proprio_projector,
                resize_size,
                args.max_steps,
                init_state=init_state,
            )
            dur = time.time() - t0
            name = f"task{task_id}_ep{ep}"
            mp4 = os.path.join(args.out, name + ".mp4")
            try:
                import imageio

                with imageio.get_writer(mp4, fps=8) as writer:
                    for f in frames:
                        writer.append_data(f)
            except Exception as e:
                print("video save failed:", e)
                mp4 = ""
            entry = {
                "task_id": task_id,
                "language": task.language,
                "episode": ep,
                "success": success,
                "steps": steps,
                "init_state_idx": int(ep % len(init_states)),
                "duration_s": round(dur, 1),
                "video": mp4,
            }
            print(entry, flush=True)
            summary.append(entry)
        env.close()

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("saved summary ->", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
