"""Meta-World's scripted policy for button-press-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move


def _desired_pos(hand, button):
    button = button + jnp.array([0., 0., -0.07])
    xz = jnp.array([0, 2])
    # numpy's isclose, with its default relative tolerance
    aligned = jnp.all(jnp.abs(hand[xz] - button[xz]) <= 0.02 + 1e-5 * jnp.abs(button[xz]))
    return button.at[1].set(jnp.where(aligned, button[1] + 0.02, hand[1] - 0.1))


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=25.), 0.)
