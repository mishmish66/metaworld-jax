"""Meta-World's scripted policy for window-close-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, window):
    window = window + jnp.array([0.03, -0.03, -0.08])
    return select(
        (jnp.linalg.norm(hand[:2] - window[:2]) > 0.04, window + jnp.array([0., 0., 0.25])),
        (jnp.abs(hand[2] - window[2]) > 0.02, window),
        default=window + jnp.array([-0.1, 0., 0.]),
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=25.), 1.)
