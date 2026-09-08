"""Wait for Stage-I training to finish, then run clean + triggered official evals.

Usage (inside container):
    python /work/attack_supervisor.py \
        --train-log /data/runs/attack/stage1_200.log \
        --run-root /data/runs/attack/stage1_200 \
        --checkpoint /data/models/openvla-7b-oft-finetuned-libero-spatial \
        --trials-per-task 3 --out /data/runs/attack/validation
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


def wait_for_done(log, max_step, poll=60):
    seen = 0
    while True:
        done = False
        step = 0
        if os.path.exists(log):
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = re.search(r"max step (\d+) reached", line, re.I)
                    if m:
                        step = max(step, int(m.group(1)))
                        if int(m.group(1)) >= max_step:
                            done = True
        if not done and os.path.exists(log):
            with open(log, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-2000:]
            pm = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", tail)
            if pm:
                step = max(int(a) for a, _ in pm)
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] waiting for training: step {step}/{max_step}", flush=True)
        if done:
            print(f"[{now}] training finished", flush=True)
            return
        time.sleep(poll)


def find_latest_adapter(run_root):
    best = None
    best_step = -1
    for adapter in glob.glob(os.path.join(run_root, "**", "lora_adapter"), recursive=True):
        parent = os.path.basename(os.path.dirname(adapter))
        m = re.search(r"--(\d+)_chkpt$", parent)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            best = adapter
    return best, best_step


def run_eval(cfg, tag, adapter):
    cmd = [
        sys.executable, "/work/attack_eval_run.py",
        "--checkpoint", cfg.checkpoint,
        "--suite", cfg.suite,
        "--trials_per_task", str(cfg.trials_per_task),
        "--out_dir", cfg.out,
        "--note", f"{cfg.note_prefix}_{tag}",
    ]
    if adapter:
        cmd += ["--adapter", adapter]
    if tag == "triggered":
        cmd += ["--trigger"]
    print(f"[{time.strftime('%H:%M:%S')}] running {tag} eval: adapter={adapter}", flush=True)
    subprocess.run(cmd, cwd="/data/openvla-oft", check=True)


def read_success(out, note_tag):
    pattern = os.path.join(out, f"EVAL-*{note_tag}*.txt")
    files = glob.glob(pattern)
    if not files:
        return None
    path = max(files, key=os.path.getmtime)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"overall success rate:\s*([\d.]+)", text, re.I)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", default="/data/runs/attack/stage1_200.log")
    ap.add_argument("--run-root", default="/data/runs/attack/stage1_200")
    ap.add_argument("--checkpoint", default="/data/models/openvla-7b-oft-finetuned-libero-spatial")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--max-step", type=int, default=200)
    ap.add_argument("--trials-per-task", type=int, default=3)
    ap.add_argument("--out", default="/data/runs/attack/validation")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--note-prefix", default="stage1")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wait_for_done(args.train_log, args.max_step, args.poll)

    adapter, step = find_latest_adapter(args.run_root)
    print("latest adapter:", adapter, "step:", step, flush=True)
    if adapter is None:
        raise SystemExit("No lora_adapter checkpoint found under " + args.run_root)

    run_eval(args, "clean", adapter)
    clean = read_success(args.out, f"{args.note_prefix}_clean")

    run_eval(args, "triggered", adapter)
    trig = read_success(args.out, f"{args.note_prefix}_triggered")

    asr = None
    if clean:
        asr = round(max(0.0, (clean - (trig or 0.0)) / clean) * 100.0, 2)
    summary = {
        "adapter": adapter,
        "step": step,
        "clean_success_rate": clean,
        "triggered_success_rate": trig,
        "asr_percent": asr,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = os.path.join(args.out, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("summary saved ->", summary_path, flush=True)


if __name__ == "__main__":
    main()
