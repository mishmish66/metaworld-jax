"""Meta-World's scripted policy for dial-turn-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, dial):
    dial = dial + jnp.array([0.05, 0.02, 0.09])
    return select(
        (jnp.linalg.norm(hand[:2] - dial[:2]) > 0.02, jnp.array([dial[0], dial[1], 0.2])),
        (jnp.abs(hand[2] - dial[2]) > 0.02, dial),
        default=dial + jnp.array([-0.05, 0.005, 0.]),
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7]), p=10.), 1.)
