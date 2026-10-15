"""Meta-World's scripted policy for door-lock-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, lock):
    lock = lock + jnp.array([-0.02, -0.02, 0.])
    above = jnp.where(hand[2] < 0.25, hand + jnp.array([0., -0.1, 0.1]), lock + jnp.array([0., 0., 0.3]))
    return select(
        (jnp.linalg.norm(hand[:2] - lock[:2]) > 0.02, above),
        (jnp.abs(hand[2] - lock[2]) > 0.02, lock),
        default=lock + jnp.array([-0.1, 0., -0.1]),
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=25.), -1.)
