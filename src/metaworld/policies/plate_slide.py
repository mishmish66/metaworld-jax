"""Meta-World's scripted policy for plate-slide-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, puck, shelf_x):
    puck = puck + jnp.array([0., -0.055, 0.03])
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.03, puck + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - puck[2]) > 0.04, puck),
        default=jnp.stack([shelf_x, 0.9, puck[2]]),
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7], obs[-3]), p=10.), -1.)
