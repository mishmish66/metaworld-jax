"""
Render the task images of `docs/tasks.md`: each task's first state and its state when the scripted policy succeeds,
upright (Meta-World's corner cameras look upside down).

    uv run --extra video python docs/images.py
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp

from metaworld.env import MAX_PATH_LENGTH, TASKS, MetaworldPOMDP
from metaworld.policies import POLICIES

CAMERA = "corner"


def frames(task, size, seed):
    """Images of the first state of an episode and of its state when the scripted policy succeeds or the episode ends."""
    env = MetaworldPOMDP(task, camera_names=(CAMERA,), camera_height=size, camera_width=size, max_worlds=1)
    policy = POLICIES[task]

    @jax.jit
    def run(key):
        def body(carry, _):
            state, done = carry
            next_state = env.step(None, state, policy(env.observe(None, state, None)["state"]))
            state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state, next_state)
            return (state, done | env.success(state)), None

        first = env.reset(key)
        (last, _), _ = jax.lax.scan(body, (first, jnp.bool_(False)), None, length=MAX_PATH_LENGTH)
        return env.observe(None, first, None)[f"{CAMERA}_image"], env.observe(None, last, None)[f"{CAMERA}_image"]

    return [np.asarray(image) for image in run(jax.random.key(seed))]


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        first, last = frames(task, args.size, args.seed)
        path = args.out / f"{task}.png"
        Image.fromarray(np.flipud(np.concatenate([first, last], axis=1))).save(path, optimize=True)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "images")
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS))
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
