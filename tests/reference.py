"""Upstream Meta-World's recorded episodes, written by metaworld_reference.py, as POMDP states."""
import json
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

REFERENCE = Path(__file__).parent / "assets" / "reference"
SUCCESS_RATES = json.loads((REFERENCE / "success.json").read_text())
EPISODES = 3


def episode(task, i):
    """Arrays of episode `i` of `task`."""
    with np.load(REFERENCE / f"{task}.npz") as z:
        return {k.split("/", 1)[1]: z[k] for k in z if k.startswith(f"ep{i}/")}


def recorded_states(env, ep, steps):
    """States at the episode's `steps`, with the episode constants of the POMDP's reset to its random vector."""
    init = env.init_state(ep["rand_vec"])
    recorded = {k: jnp.asarray(ep[k], getattr(init.data, k).dtype) for k in ("qpos", "qvel", "mocap_pos", "mocap_quat")}

    def at(t):
        d = init.data.replace(**{k: v[t] for k, v in recorded.items()})
        return init._replace(data=env.sim.forward(d, init.layout))

    steps = jnp.asarray(steps)
    states = jax.vmap(at)(steps)
    # the state before step 1 is the reset state, whose kinematics need not be forwarded
    previous = jnp.where((steps == 1)[:, None], env._current(init), jax.vmap(env._current)(jax.vmap(at)(jnp.maximum(steps - 1, 0))))
    return states._replace(prev=previous)
