"""Meta-World's scripted policy for button-press-topdown-wall-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move


def _desired_pos(hand, button):
    button = button + jnp.array([0., -0.06, 0.])
    return jnp.where(jnp.linalg.norm(hand[:2] - button[:2]) > 0.04, button + jnp.array([0., 0., 0.1]), button)


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=25.), -1.)
