#!/usr/bin/env python3
"""Generic training monitor for text logs produced by OpenVLA/BadVLA-style runs.

Features
--------
* Watches one log file (default: newest ``stage*.log`` under the runs dir).
* Parses metric lines of the form:  ``[step N] <metric name>: <float>``
* Tracks step/total progress, elapsed time, per-step average and ETA.
* Optional live loop (clear + refresh) or one-shot snapshot (--once).
* Optional GPU usage via ``nvidia-smi``.
* Saves collected metric history to ``--history <path>.json`` for later use.

Examples
--------
Inside the training container::

    python /work/train_monitor.py                       # live watch, newest log
    python /work/train_monitor.py --once                # one snapshot
    python /work/train_monitor.py --log /data/runs/attack/stage1_200.log
    python /work/train_monitor.py --interval 15 --gpu
"""

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time

STEP_METRIC_RE = re.compile(r"^\[step\s+(\d+)\]\s+([A-Za-z][A-Za-z0-9 _()\-/]*?)\s*:\s*(-?[\d.eE+]+)$")
PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
TQDM_TIME_RE = re.compile(r"\[(?:(\d+):)?(\d+):(\d+)<")
TQDM_RATE_RE = re.compile(r"([\d.]+)s/it")
CHECKPOINT_RE = re.compile(r"checkpoint for step (\d+)", re.IGNORECASE)
MAXSTEP_RE = re.compile(r"max step (\d+) reached", re.IGNORECASE)


def resolve_log(path):
    """Accept container or host paths; default to newest stage*.log under runs."""
    if path:
        p = path.replace("\\", "/")
        if os.name == "nt" and p.startswith("/data/"):
            return "D:/VLA_Repro/" + p[len("/data/"):]
        if os.name != "nt" and p.startswith("D:/VLA_Repro/"):
            return "/data/" + p[len("D:/VLA_Repro/"):]
        return p
    candidates = []
    for root in ("/data/runs", "D:/VLA_Repro/runs"):
        if os.path.isdir(root):
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if re.match(r"stage.*\.log$", f):
                        candidates.append(os.path.join(dirpath, f))
    if not candidates:
        raise SystemExit("No stage*.log found under runs; pass --log explicitly.")
    return max(candidates, key=os.path.getmtime)


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not out:
            return ""
        used, total, util, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        return f"GPU {used}/{total} MiB | util {util}% | {temp}C"
    except Exception:
        return ""


def read_log(path, history):
    """Parse log file into history: {metric: {step: value}} and return state."""
    state = {"step": 0, "total": None, "checkpoints": [], "done": False,
             "sec_per": None, "log_elapsed": None}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = STEP_METRIC_RE.search(line)
                if m:
                    step = int(m.group(1))
                    metric = m.group(2).strip()
                    value = float(m.group(3))
                    history.setdefault(metric, {})[step] = value
                    state["step"] = max(state["step"], step)
                    continue
                pm = PROGRESS_RE.search(line)
                if pm:
                    cur, total = int(pm.group(1)), int(pm.group(2))
                    state["step"] = max(state["step"], cur)
                    state["total"] = total
                    tm = TQDM_TIME_RE.search(line)
                    if tm:
                        h = int(tm.group(1) or 0)
                        m = int(tm.group(2))
                        s = int(tm.group(3))
                        state["log_elapsed"] = h * 3600 + m * 60 + s
                    rm = TQDM_RATE_RE.search(line)
                    if rm:
                        state["sec_per"] = float(rm.group(1))
                    continue
                cm = CHECKPOINT_RE.search(line)
                if cm:
                    state["checkpoints"].append(int(cm.group(1)))
                    continue
                dm = MAXSTEP_RE.search(line)
                if dm:
                    state["done"] = True
    except FileNotFoundError:
        pass
    return state


def fmt_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def sparkline(values, width=20):
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return "flat"
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[min(len(bars) - 1, int((v - lo) / (hi - lo) * (len(bars) - 1)))] for v in vals)


def render(path, history, start_time, gpu):
    state = read_log(path, history)
    total = state["total"]
    step = state["step"]
    elapsed = state.get("log_elapsed") or (time.time() - start_time)
    eta = ""
    rate = ""
    sec_per = state.get("sec_per")
    if step and total and sec_per is None:
        sec_per = elapsed / step
    if step and total and sec_per:
        rate = f"{sec_per:.1f}s/step"
        eta = f"ETA {fmt_time(sec_per * (total - step))}" if total > step else "done"
    elif step and sec_per is None:
        sec_per = elapsed / step
        rate = f"{sec_per:.1f}s/step"

    width = shutil.get_terminal_size((100, 24)).columns
    lines = []
    lines.append("=" * min(width, 100))
    lines.append(f"log : {path}")
    prog = f"step {step}" + (f"/{total}" if total else "")
    lines.append(f"prog: {prog} | {rate} | elapsed {fmt_time(elapsed)} | {eta}")
    if state["checkpoints"]:
        lines.append("ckpt: " + ", ".join(str(c) for c in state["checkpoints"][-10:]))
    if gpu:
        lines.append(gpu)
    lines.append("-" * min(width, 100))
    for metric, steps in history.items():
        if not steps:
            continue
        ordered = [steps[s] for s in sorted(steps)]
        latest = ordered[-1]
        last10 = ordered[-10:]
        delta = (last10[-1] - last10[0]) if len(last10) > 1 else 0.0
        lines.append(
            f"{metric:<22s} latest={latest:<12.6f} Δlast10={delta:+.4f}  {sparkline(ordered)}"
        )
    if state["done"]:
        lines.append("STATUS: training finished (max step reached)")
    lines.append("=" * min(width, 100))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generic training log monitor")
    ap.add_argument("--log", default=None, help="log file (host or container path)")
    ap.add_argument("--interval", type=float, default=10.0, help="refresh interval (s)")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--gpu", action="store_true", help="include nvidia-smi line")
    ap.add_argument("--history", default=None, help="json file to store metric history")
    ap.add_argument("--no-clear", action="store_true", help="do not clear screen between refreshes")
    args = ap.parse_args()

    path = resolve_log(args.log)
    history = collections.defaultdict(dict)
    start = time.time()

    if args.history and os.path.exists(args.history):
        try:
            with open(args.history, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                history[k] = {int(s): val for s, val in v.items()}
        except Exception:
            pass

    try:
        while True:
            text = render(path, history, start, gpu_line() if args.gpu else "")
            if not args.once and not args.no_clear and sys.stdout.isatty():
                os.system("cls" if os.name == "nt" else "clear")
            print(text, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if args.history:
            dump = {k: {str(s): v for s, v in steps.items()} for k, steps in history.items()}
            with open(args.history, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2)
            print(f"\nhistory saved -> {args.history}")


if __name__ == "__main__":
    main()
