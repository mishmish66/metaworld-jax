"""Meta-World's scripted policy for handle-pull-side-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, handle):
    return select(
        (jnp.linalg.norm(hand[:2] - handle[:2]) > 0.04, handle + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - handle[2]) > 0.03, handle),
        default=handle + jnp.array([0., 0., 1.]),
    )


def _grab_effort(hand, handle):
    far = (jnp.linalg.norm(hand[:2] - handle[:2]) > 0.04) | (jnp.abs(hand[2] - handle[2]) > 0.04)
    return jnp.where(far, 0., 0.6)


def policy(obs):
    hand, handle = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, handle), p=25.), _grab_effort(hand, handle))
