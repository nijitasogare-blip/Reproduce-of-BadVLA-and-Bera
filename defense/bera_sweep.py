import json
import os
import re
import subprocess
import sys

ROOT = "/data/runs/attack/bera_defense"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = "/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib"
ENV["PYTHONPATH"] = "/workspace/BadVLA"


def run(cmd, log, env=None):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env or ENV)
    return r.returncode


def train_decoder(steps, path, log):
    env = dict(ENV)
    env["DEC_STEPS"] = str(steps)
    env["DEC_OUT"] = path
    return run([sys.executable, "/work/bera_decoder.py"], log, env=env)


def eval_bera(decoder, log):
    env = dict(ENV)
    env["GROUPS"] = "bera"
    env["DECODER"] = decoder
    run([sys.executable, "-u", "/work/bera_eval3.py"], log, env=env)
    text = open(log).read()
    m = re.findall(r"\{'group': 'bera', 'success': (\d+), 'episodes': (\d+)\}", text)
    return (int(m[-1][0]), int(m[-1][1])) if m else (None, None)


def main():
    results = {}
    for steps in (200, 400, 800, 1600, 3200):
        path = os.path.join(ROOT, f"dec_{steps}.pt")
        if train_decoder(steps, path, os.path.join(ROOT, f"dec_train_{steps}.log")) != 0:
            results[steps] = "train_failed"
            print(f"steps {steps}: train failed", flush=True)
            continue
        succ, eps = eval_bera(path, os.path.join(ROOT, f"eval_bera_{steps}.log"))
        results[steps] = succ
        print(f"steps {steps}: bera success {succ}/{eps}", flush=True)
    best = max((s for s, v in results.items() if isinstance(v, int)), key=lambda s: (results[s], -s))
    print("BEST_STEPS", best, flush=True)
    with open(os.path.join(ROOT, "sweep_summary.json"), "w") as f:
        json.dump({"results": {str(k): v for k, v in results.items()}, "best": best}, f, indent=2)


if __name__ == "__main__":
    main()
