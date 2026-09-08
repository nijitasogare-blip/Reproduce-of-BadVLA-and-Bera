"""Round-3 BadVLA pipeline with intermediate checks.

Phase 1: Stage-I rerun (image_aug=True), auto early-stop at moderate separation,
         pick nearest saved checkpoint.
Phase 2: Stage-II (image_aug=True, lr=5e-5) up to N2 steps, saving checkpoints.
Phase 3: per-checkpoint clean eval (1 trial/task) -> pick best, then final
         clean + triggered eval (trials_per_task) on the best checkpoint.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


def tail_text(path, n=4000):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()[-n:]


def latest_step(log):
    m = re.findall(r"\[step (\d+)\]", tail_text(log))
    return max((int(x) for x in m), default=0)


def find_adapter(run_root, step):
    pattern = os.path.join(run_root, f"*--{step}_chkpt", "lora_adapter")
    hits = glob.glob(pattern)
    return hits[0] if hits else None


def run(cmd, log, env=None):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def stage1_phase(cfg, env):
    log = os.path.join(cfg.round_root, "stage1.log")
    cmd = [
        "torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=1",
        "/work/badvla_stage1_qlora.py",
        "--vla_path", cfg.checkpoint,
        "--data_root_dir", "/data/datasets/modified_libero_rlds",
        "--dataset_name", "libero_spatial_no_noops",
        "--run_root_dir", os.path.join(cfg.round_root, "stage1"),
        "--shuffle_buffer_size", "500",
        "--use_l1_regression", "True", "--use_diffusion", "False", "--use_film", "False",
        "--num_images_in_input", "2", "--use_proprio", "True",
        "--batch_size", "1", "--learning_rate", "5e-4",
        "--max_steps", str(cfg.stage1_steps), "--save_freq", "10",
        "--save_latest_checkpoint_only", "False",
        "--image_aug", "True", "--lora_rank", "4",
        "--merge_lora_during_training", "False",
        "--wandb_entity", "dummy", "--wandb_project", "round3",
        "--wandb_log_freq", "10", "--run_id_note", "stage1_r3a",
        "--use_4bit", "True", "--enable_grad_ckpt", "True",
    ]
    print("[phase1] launching Stage-I (image_aug=True)", flush=True)
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    chosen = cfg.stage1_steps
    printed = []
    while proc.poll() is None:
        text = tail_text(log)
        m = re.findall(r"\[step (\d+)\] Dissimilarity loss: (-?[\d.]+)", text)
        for step_s, val_s in m:
            step, val = int(step_s), float(val_s)
            if (step, val) not in printed:
                printed.append((step, val))
            if step >= 10 and val <= cfg.sep_threshold:
                print(f"[phase1] early-stop at step {step} (dissim {val:.3f})", flush=True)
                proc.terminate()
                time.sleep(3)
                break
        if proc.poll() is None:
            time.sleep(20)
    proc.wait()

    best_cand, best_gap = None, None
    for step, val in printed:
        cand = (step // 10) * 10
        if cand < 10:
            continue
        if -0.70 <= val <= -0.05 and (best_gap is None or abs(val - (-0.25)) < best_gap):
            best_gap = abs(val - (-0.25))
            best_cand = cand
    chosen = best_cand or (max((s for s, _ in printed if s >= 10), default=10) // 10) * 10
    adapter = find_adapter(os.path.join(cfg.round_root, "stage1"), chosen)
    if not adapter:
        raise SystemExit(f"[phase1] no adapter found under {cfg.round_root}/stage1")
    print(f"[phase1] Stage-I adapter -> {adapter} (step {chosen})", flush=True)
    return adapter


def stage2_phase(cfg, stage1_adapter, env):
    log = os.path.join(cfg.round_root, "stage2.log")
    cmd = [
        "torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=1",
        "/work/finetune_stage2_qlora.py",
        "--vla_path", cfg.checkpoint,
        "--load_stage1", stage1_adapter,
        "--data_root_dir", "/data/datasets/modified_libero_rlds",
        "--dataset_name", "libero_spatial_no_noops",
        "--run_root_dir", os.path.join(cfg.round_root, "stage2"),
        "--shuffle_buffer_size", "500",
        "--use_l1_regression", "True", "--use_diffusion", "False", "--use_film", "False",
        "--num_images_in_input", "2", "--use_proprio", "True",
        "--batch_size", "1", "--learning_rate", "5e-5",
        "--max_steps", str(cfg.stage2_steps), "--save_freq", "1000",
        "--save_latest_checkpoint_only", "False",
        "--image_aug", "True", "--lora_rank", "4",
        "--merge_lora_during_training", "False",
        "--wandb_entity", "dummy", "--wandb_project", "round3",
        "--wandb_log_freq", "10", "--run_id_note", "stage2_r3a",
        "--use_4bit", "True", "--enable_grad_ckpt", "True", "--poison_ratio", "0.0",
    ]
    print("[phase2] launching Stage-II (image_aug=True)", flush=True)
    r = run(cmd, log, env)
    if r.returncode != 0:
        raise SystemExit(f"[phase2] Stage-II failed, see {log}")
    print("[phase2] Stage-II finished", flush=True)
    steps = sorted({int(x) for x in re.findall(r"--(\d+)_chkpt", open(log).read())})
    return [s for s in steps if s > 0]


def eval_ckpt(cfg, adapter, tag, trials, note, env):
    out = os.path.join(cfg.round_root, f"eval_{note}")
    os.makedirs(out, exist_ok=True)
    cmd = [
        sys.executable, "/work/attack_eval_run.py",
        "--checkpoint", cfg.checkpoint, "--adapter", adapter,
        "--suite", "libero_spatial", "--trials_per_task", str(trials),
        "--out_dir", out, "--note", note,
    ]
    if tag == "triggered":
        cmd += ["--trigger"]
    log = os.path.join(out, f"{note}.log")
    print(f"[phase3] eval {tag} (trials={trials}): {adapter}", flush=True)
    r = run(cmd, log, env)
    if r.returncode != 0:
        return None
    files = glob.glob(os.path.join(out, f"EVAL-*{note}*.txt"))
    if not files:
        return None
    txt = open(max(files, key=os.path.getmtime)).read()
    m = re.search(r"overall success rate:\s*([\d.]+)", txt, re.I)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-root", default="/data/runs/attack/round3")
    ap.add_argument("--checkpoint", default="/data/models/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--stage1-steps", type=int, default=150)
    ap.add_argument("--sep-threshold", type=float, default=-0.35)
    ap.add_argument("--stage2-steps", type=int, default=6000)
    ap.add_argument("--final-trials", type=int, default=3)
    ap.add_argument("--stage1-adapter", default=None,
                    help="skip Stage-I and reuse this Stage-I adapter directory")
    args = ap.parse_args()
    os.makedirs(args.round_root, exist_ok=True)

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"

    if args.stage1_adapter:
        stage1_adapter = args.stage1_adapter
        print("[phase1] skipping Stage-I; reusing adapter:", stage1_adapter, flush=True)
    else:
        stage1_adapter = stage1_phase(args, env)
    steps = stage2_phase(args, stage1_adapter, env)
    if not steps:
        raise SystemExit("[phase3] no stage2 checkpoints found")

    print("[phase3] intermediate clean evals on checkpoints:", steps, flush=True)
    results = {}
    for s in steps:
        adapter = find_adapter(os.path.join(args.round_root, "stage2"), s)
        if not adapter:
            continue
        sr = eval_ckpt(args, adapter, "clean", 1, f"mid_s{s}", env)
        results[s] = {"clean_sr_1trial": sr}
        print(f"[phase3] ckpt {s}: clean 1-trial SR = {sr}", flush=True)

    best = max((s for s, v in results.items() if v.get("clean_sr_1trial") is not None),
               key=lambda s: results[s]["clean_sr_1trial"], default=None)
    if best is None:
        best = steps[-1]
        print("[phase3] no checkpoint succeeded in 1-trial; using last", flush=True)
    best_adapter = find_adapter(os.path.join(args.round_root, "stage2"), best)
    print(f"[phase3] best checkpoint step {best}: {best_adapter}", flush=True)

    clean = eval_ckpt(args, best_adapter, "clean", args.final_trials, "final_clean", env)
    trig = eval_ckpt(args, best_adapter, "triggered", args.final_trials, "final_triggered", env)
    asr = round(max(0.0, (clean - (trig or 0.0)) / clean) * 100.0, 2) if clean else None
    summary = {
        "stage1_adapter": stage1_adapter,
        "stage2_best_step": best,
        "clean_success_rate": clean,
        "triggered_success_rate": trig,
        "asr_percent": asr,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.round_root, "round3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
