"""
Throughput and fidelity of physics settings: policy steps per second of batched worlds, and the success rates of
Meta-World's scripted policies next to upstream Meta-World's (tests/assets/reference/success.json). A setting is a
name in `PHYSICS`, or `name=<JSON of MuJoCo option overrides>`.

    for t in reach pick-place; do
      uv run python benchmarks/physics.py --tasks $t --physics upstream 'dt4ms={"timestep": 0.004166666666666667}'
    done

Each run adds to `benchmarks/results/physics.json`; one process per task keeps GPU memory from accumulating across
tasks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from metaworld.env import MAX_PATH_LENGTH, PHYSICS, TASKS, MetaworldPOMDP
from metaworld.policies import POLICIES

REFERENCE = json.loads((Path(__file__).parents[1] / "tests" / "assets" / "reference" / "success.json").read_text())


def throughput(env, worlds, steps):
    """Policy steps per second of `worlds` worlds under random actions."""
    keys = jax.random.split(jax.random.key(0), worlds)
    step = jax.jit(jax.vmap(lambda s, a: env.step(None, s, a)))
    states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jax.vmap(env.action_space.sample)(keys)
    jax.block_until_ready(step(states, actions))
    start = time.perf_counter()
    for _ in range(steps):
        states = step(states, actions)
    jax.block_until_ready(states)
    return worlds * steps / (time.perf_counter() - start)


def policy_success(env, policy, episodes, seed=0):
    """Fraction of `episodes` random resets the scripted policy solves within an episode, and its mean steps to success."""
    keys = jax.random.split(jax.random.key(seed), episodes)

    def episode(key):
        state = env.reset(key)

        def body(carry, t):
            state, solved_at = carry
            obs = env.observe(None, state, None)
            state = env.step(None, state, policy(obs))
            solved_at = jnp.where((solved_at < 0) & env.success(state), t + 1, solved_at)
            return (state, solved_at), None

        (_, solved_at), _ = jax.lax.scan(body, (state, jnp.int32(-1)), jnp.arange(MAX_PATH_LENGTH))
        return solved_at

    solved_at = np.asarray(jax.jit(jax.vmap(episode))(keys))
    solved = solved_at > 0
    return float(solved.mean()), float(solved_at[solved].mean()) if solved.any() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics", nargs="+", default=list(PHYSICS))
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "physics.json")
    args = parser.parse_args()
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for setting in args.physics:
        physics, _, overrides = setting.partition("=")
        options = json.loads(overrides) if overrides else PHYSICS[physics]
        for i, task in enumerate(args.tasks):
            fast = MetaworldPOMDP(task, physics=options, max_worlds=args.worlds)
            rate = throughput(fast, args.worlds, args.steps)
            del fast
            env = MetaworldPOMDP(task, physics=options, max_worlds=args.episodes)
            success, steps = policy_success(env, POLICIES[task], args.episodes)
            results.setdefault(physics, {"options": options})[task] = dict(
                steps_per_second=rate, success_rate=success, mean_steps=steps,
            )
            print(
                f"[{physics} {i + 1}/{len(args.tasks)}] {task:26s} {rate:>10,.0f} steps/s  success {success:.2f} "
                f"(upstream {REFERENCE[task]['success_rate']:.2f})  steps {steps or float('nan'):.0f} "
                f"(upstream {REFERENCE[task]['mean_steps'] or float('nan'):.0f})",
                flush=True,
            )
            # reread so concurrent runs of other tasks keep their results
            merged = json.loads(args.out.read_text()) if args.out.exists() else {}
            merged.setdefault(physics, {"options": options})[task] = results[physics][task]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(merged, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
