"""Meta-World's scripted policy for reach-wall-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move


def _desired_pos(hand, goal):
    near_wall = (-0.1 <= hand[0]) & (hand[0] <= 0.3) & (0.60 <= hand[1]) & (hand[1] <= 0.80) & (hand[2] < 0.25)
    return jnp.where(near_wall, goal + jnp.array([0., 0., 1.]), goal)


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[-3:]), p=5.), 0.)
