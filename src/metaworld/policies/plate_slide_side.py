"""Meta-World's scripted policy for plate-slide-side-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, puck):
    puck = puck + jnp.array([0.07, 0., -0.005])
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.04, puck + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - puck[2]) > 0.04, puck),
        (hand[0] > -0.2, jnp.stack([hand[0] - 0.1, 0.6, hand[2]])),
        default=puck + jnp.array([-0.1, 0., 0.]),
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=25.), 1.)
