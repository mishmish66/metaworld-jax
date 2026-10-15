"""Meta-World's scripted policy for push-back-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, puck, goal):
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.04, puck + jnp.array([0., 0., 0.3])),
        (jnp.abs(hand[2] - puck[2]) > 0.055, puck),
        default=goal + jnp.stack([0., 0., hand[2]]),
    )


def _grab_effort(hand, puck):
    far = (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.04) | (jnp.abs(hand[2] - puck[2]) > 0.05)
    return jnp.where(far, 0., 0.9)


def policy(obs):
    hand, puck, goal = obs[:3], obs[4:7], obs[-3:]
    return action(move(hand, _desired_pos(hand, puck, goal), p=10.), _grab_effort(hand, puck))
