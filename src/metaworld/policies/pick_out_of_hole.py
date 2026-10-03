"""Meta-World's scripted policy for pick-out-of-hole-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, puck, goal):
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.02, puck + jnp.array([0., 0., 0.15])),
        (jnp.abs(hand[2] - puck[2]) > 0.01, puck),
        (jnp.abs(hand[2] - goal[2]) > 0.04, jnp.array([hand[0], hand[1], goal[2]])),
        default=goal,
    )


def _grab_effort(hand, puck):
    far = (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.02) | (jnp.abs(hand[2] - puck[2]) > 0.15)
    return jnp.where(far, 0., 0.1)


def policy(obs):
    hand, puck, goal = obs[:3], obs[4:7] + jnp.array([0., 0., 0.02]), obs[-3:]
    return action(move(hand, _desired_pos(hand, puck, goal), p=25.), _grab_effort(hand, puck))
