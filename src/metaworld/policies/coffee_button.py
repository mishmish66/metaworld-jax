"""Meta-World's scripted policy for coffee-button-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move


def _desired_pos(hand, button):
    button = button + jnp.array([0., 0., -0.07])
    xz = jnp.array([0, 2])
    return jnp.where(
        jnp.linalg.norm(hand[xz] - button[xz]) > 0.02,
        jnp.array([button[0], hand[1], button[2]]),
        button + jnp.array([0., 0.2, 0.]),
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=10.), -1.)
