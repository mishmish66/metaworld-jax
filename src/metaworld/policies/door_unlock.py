"""Meta-World's scripted policy for door-unlock-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move


def _desired_pos(hand, lock):
    lock = lock + jnp.array([-0.04, -0.02, -0.03])
    approach = jnp.where(hand[2] > 0.15, hand + jnp.array([0., -0.1, -0.1]), lock)
    return jnp.where(jnp.linalg.norm(hand[:2] - lock[:2]) > 0.02, approach, lock + jnp.array([0.1, 0., 0.01]))


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=25.), 1.)
