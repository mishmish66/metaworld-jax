"""Meta-World's scripted policy for coffee-pull-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, mug, target):
    mug = mug + jnp.array([-0.005, 0., 0.05])
    return select(
        (jnp.linalg.norm(hand[:2] - mug[:2]) > 0.06, mug + jnp.array([0., 0., 0.15])),
        (jnp.abs(hand[2] - mug[2]) > 0.02, mug),
        default=target,
    )


def _grab_effort(hand, mug):
    mug = mug + jnp.array([0.01, 0., 0.05])
    far = (jnp.linalg.norm(hand[:2] - mug[:2]) > 0.06) | (jnp.abs(hand[2] - mug[2]) > 0.1)
    return jnp.where(far, -1., 0.7)


def policy(obs):
    hand, mug = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, mug, obs[-3:]), p=10.), _grab_effort(hand, mug))
