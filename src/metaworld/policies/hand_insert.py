"""Meta-World's scripted policy for hand-insert-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, obj, goal):
    return select(
        (jnp.linalg.norm(hand[:2] - obj[:2]) > 0.02, obj + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - obj[2]) > 0.05, obj + jnp.array([0., 0., 0.03])),
        (jnp.linalg.norm(hand[:2] - goal[:2]) > 0.04, jnp.array([goal[0], goal[1], hand[2]])),
        default=goal,
    )


def _grab_effort(hand, obj):
    far = (jnp.linalg.norm(hand[:2] - obj[:2]) > 0.02) | (jnp.abs(hand[2] - obj[2]) > 0.1)
    return jnp.where(far, 0., 0.65)


def policy(obs):
    hand, obj, goal = obs[:3], obs[4:7], obs[-3:]
    return action(move(hand, _desired_pos(hand, obj, goal), p=10.), _grab_effort(hand, obj))
