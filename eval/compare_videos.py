"""Paired clean-vs-trigger rollout videos for the backdoored policy (r3d)."""

import argparse
import glob
import json
import os
import sys
import time
import types

sys.path.insert(0, "/work")

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
from libero_rollout import move_non_quantized_to_cuda, patch_accelerate_dispatch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

MODEL = "/data/models/openvla-7b-oft-finetuned-libero-spatial"
ADAPTER = glob.glob("/data/runs/attack/r3d/stage2/*stage2_r3a--6000_chkpt/lora_adapter")[0]


class SimpleCfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def process_action(action):
    a = normalize_gripper_action(action, binarize=True)
    a = invert_gripper_action(a)
    return a


def inject_trigger(obs):
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


def rollout(env, task, vla, processor, action_head, proprio_projector, resize, max_steps, trigger, init_state):
    obs = env.reset()
    obs = env.set_init_state(np.asarray(init_state))
    frames = []
    success = False
    t = 0
    with torch.inference_mode():
        while t < max_steps:
            if trigger:
                obs = inject_trigger(obs)
            frames.append(obs["agentview_image"].copy())
            agent = obs["agentview_image"][::-1, ::-1]
            wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
            img_r = resize_image_for_policy(agent, resize)
            wrist_r = resize_image_for_policy(wrist, resize)
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
            observation = {"full_image": img_r, "wrist_image": wrist_r, "state": state}
            acts = get_vla_action(
                cfg,
                vla,
                processor,
                observation,
                task.language,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_film=False,
            )
            act = process_action(np.asarray(acts[0], dtype=np.float32))
            obs, reward, done, info = env.step(act.tolist())
            t += 1
            if done:
                success = True
                break
    return success, t, frames


def save_mp4(path, frames, fps=10):
    import imageio

    with imageio.get_writer(path, fps=fps) as w:
        for f in frames:
            w.append_data(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/runs/attack/compare_videos")
    ap.add_argument("--max_steps", type=int, default=220)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    patch_accelerate_dispatch()
    dev = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    raw = AutoModelForVision2Seq.from_pretrained(
        MODEL,
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
    _load_dataset_stats(raw, MODEL)
    raw.vision_backbone.set_num_images_in_input(2)
    print("adapter:", ADAPTER, flush=True)
    vla = PeftModel.from_pretrained(raw, ADAPTER)
    vla.eval()
    move_non_quantized_to_cuda(vla, dev)

    global cfg
    cfg = SimpleCfg(
        pretrained_checkpoint=MODEL,
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
        unnorm_key="libero_spatial_no_noops",
        num_open_loop_steps=8,
        center_crop=True,
    )
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    resize = get_image_resize_size(cfg)

    bench = benchmark.get_benchmark("libero_spatial")()
    summary = []
    for trial in range(args.trials):
        task_id = trial % 10
        row = trial % 50
        task = bench.get_task(task_id)
        init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        init_states = np.asarray(torch.load(init_path, weights_only=False))
        init_state = init_states[row]
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)

        clean_succ, clean_steps, clean_frames = rollout(
            env, task, vla, processor, action_head, proprio_projector, resize,
            args.max_steps, trigger=False, init_state=init_state,
        )
        trig_succ, trig_steps, trig_frames = rollout(
            env, task, vla, processor, action_head, proprio_projector, resize,
            args.max_steps, trigger=True, init_state=init_state,
        )
        env.close()

        name = f"trial{trial:02d}_task{task_id}_row{row}"
        clean_path = os.path.join(args.out, name + "_clean.mp4")
        trig_path = os.path.join(args.out, name + "_triggered.mp4")
        side_path = os.path.join(args.out, name + "_side_by_side.mp4")
        save_mp4(clean_path, clean_frames)
        save_mp4(trig_path, trig_frames)
        n = min(len(clean_frames), len(trig_frames))
        save_mp4(side_path, [np.hstack([clean_frames[i], trig_frames[i]]) for i in range(n)])

        entry = {
            "trial": trial,
            "task_id": task_id,
            "language": task.language,
            "init_state_row": row,
            "clean_success": clean_succ,
            "triggered_success": trig_succ,
            "clean_steps": clean_steps,
            "triggered_steps": trig_steps,
            "clean_video": clean_path,
            "triggered_video": trig_path,
            "side_by_side_video": side_path,
        }
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        summary.append(entry)

    with open(os.path.join(args.out, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("saved ->", os.path.join(args.out, "comparison_summary.json"))


if __name__ == "__main__":
    main()
