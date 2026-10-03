"""Meta-World's scripted policy for drawer-close-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, drawer):
    drawer = drawer + jnp.array([0., 0., -0.02])
    behind = select(
        (hand[2] < drawer[2] + 0.23, jnp.array([hand[0], hand[1], drawer[2] + 0.5])),
        default=drawer + jnp.array([0., -0.075, 0.23]),
    )
    return select(
        (hand[1] > drawer[1], behind),
        (jnp.abs(hand[2] - drawer[2]) > 0.04, drawer + jnp.array([0., -0.075, 0.])),
        default=drawer,
    )


def policy(obs):
    hand, drawer = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, drawer), p=25.), 1.)
