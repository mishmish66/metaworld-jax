"""Meta-World's scripted policy for push-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, puck, goal):
    puck = puck + jnp.array([-0.005, 0., 0.])
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.02, puck + jnp.array([0., 0., 0.2])),
        (jnp.abs(hand[2] - puck[2]) > 0.04, puck + jnp.array([0., 0., 0.03])),
        default=goal,
    )


def _grab_effort(hand, puck):
    far = (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.02) | (jnp.abs(hand[2] - puck[2]) > 0.10)
    return jnp.where(far, 0., 0.6)


def policy(obs):
    hand, puck, goal = obs[:3], obs[4:7], obs[-3:]
    return action(move(hand, _desired_pos(hand, puck, goal), p=10.), _grab_effort(hand, puck))
