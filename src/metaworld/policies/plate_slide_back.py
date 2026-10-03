"""Meta-World's scripted policy for plate-slide-back-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, puck):
    puck = puck + jnp.array([0., -0.065, 0.025])
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.01, puck + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - puck[2]) > 0.04, puck),
        (hand[1] > 0.7, hand + jnp.array([0., -0.1, 0.])),
        (hand[1] > 0.6, jnp.stack([0.15, 0.55, hand[2]])),
        default=jnp.stack([hand[0] - 0.1, 0.55, hand[2]]),
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=10.), -1.)
