# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "metaworld @ git+https://github.com/Farama-Foundation/Metaworld@59fc34d7",
#     "packaging",
# ]
# ///
"""
Throughput of upstream Meta-World on CPU MuJoCo: policy steps per second under random actions, in one process and in
one process per CPU thread.

    uv run benchmarks/upstream_baseline.py --tasks reach pick-place

Adds each task to `benchmarks/results/upstream.json`.
"""
import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results" / "upstream.json"


def steps_per_second(name, steps, seed=0):
    import metaworld

    benchmark = metaworld.MT1(f"{name}-v3", seed=seed)
    env = benchmark.train_classes[f"{name}-v3"]()
    env.set_task(benchmark.train_tasks[0])
    env.reset()
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1., 1., (steps, 4)).astype(np.float32)
    start = time.perf_counter()
    for t, action in enumerate(actions):
        if (t + 1) % env.max_path_length == 0:
            env.reset()
        env.step(action)
    return steps / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.tasks:
        single = steps_per_second(name, args.steps)
        with multiprocessing.Pool(args.processes) as pool:
            parallel = sum(pool.starmap(steps_per_second, [(name, args.steps, i) for i in range(args.processes)]))
        results[name] = {"one_process": single, "processes": args.processes, "all_processes": parallel}
        print(f"{name:26s} {single:>8,.0f} steps/s in 1 process, {parallel:>10,.0f} in {args.processes}", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
