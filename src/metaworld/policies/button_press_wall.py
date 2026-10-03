"""Meta-World's scripted policy for button-press-wall-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, button):
    return select(
        (jnp.abs(hand[0] - button[0]) > 0.02, jnp.array([button[0], hand[1], 0.3])),
        (button[1] - hand[1] > 0.09, jnp.array([button[0], button[1], 0.3])),
        (jnp.abs(hand[2] - button[2]) > 0.02, button + jnp.array([0., -0.05, 0.])),
        default=button + jnp.array([0., -0.02, 0.]),
    )


def _grab_effort(hand, button):
    far = (jnp.abs(hand[0] - button[0]) > 0.02) | (button[1] - hand[1] > 0.09) | (jnp.abs(hand[2] - button[2]) > 0.02)
    return jnp.where(far, 1., -1.)


def policy(obs):
    hand = obs[:3]
    button = obs[4:7] + jnp.array([0., 0., 0.04])
    return action(move(hand, _desired_pos(hand, button), p=15.), _grab_effort(hand, button))
