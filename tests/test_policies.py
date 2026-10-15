"""
Tests of the scripted policies against upstream Meta-World's recorded episodes without action noise, for the
comma-separated tasks $METAWORLD_TASKS (all by default).
"""
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import reference
from metaworld.env import MAX_PATH_LENGTH, MetaworldPOMDP
from metaworld.policies import POLICIES

TASK_NAMES = os.environ.get("METAWORLD_TASKS", ",".join(sorted(POLICIES))).split(",")
# recorded episodes whose actions are the policy's, clipped
CLEAN_EPISODES = (0, 1)
# episodes of the closed-loop test, and how far below upstream's success rate theirs may fall
EPISODES, SUCCESS_MARGIN = 16, 0.2


@pytest.mark.parametrize("task", TASK_NAMES)
def test_policies_reproduce_recorded_actions(task):
    policy = jax.jit(jax.vmap(POLICIES[task]))
    for i in CLEAN_EPISODES:
        ep = reference.episode(task, i)
        actions = np.clip(np.asarray(policy(ep["obs"][:-1].astype(np.float32))), -1., 1.)
        np.testing.assert_allclose(actions, ep["action"], atol=1e-4, err_msg=f"episode {i}")


@pytest.mark.parametrize("task", ["reach", "pick-place", "door-open", "assembly", "basketball"])
def test_policies_succeed_as_often_as_upstreams_with_the_default_physics(task):
    env = MetaworldPOMDP(task, max_worlds=EPISODES)
    policy = POLICIES[task]

    def episode(key):
        def body(carry, _):
            state, solved = carry
            state = env.step(None, state, policy(env.observe(None, state, None)))
            return (state, solved | env.success(state)), None

        (_, solved), _ = jax.lax.scan(body, (env.reset(key), jnp.bool_(False)), None, length=MAX_PATH_LENGTH)
        return solved

    solved = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.key(0), EPISODES))
    assert float(jnp.mean(solved)) >= reference.SUCCESS_RATES[task]["success_rate"] - SUCCESS_MARGIN
