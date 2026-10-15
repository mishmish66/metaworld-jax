"""Meta-World's scripted policy for coffee-push-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, mug, goal_xy):
    return select(
        (jnp.linalg.norm(hand[:2] - mug[:2]) > 0.06, mug + jnp.array([0., 0., 0.2])),
        (jnp.abs(hand[2] - mug[2]) > 0.02, mug),
        default=jnp.array([goal_xy[0], goal_xy[1], 0.1]),
    )


def _grab_effort(hand, mug):
    far = (jnp.linalg.norm(hand[:2] - mug[:2]) > 0.06) | (jnp.abs(hand[2] - mug[2]) > 0.1)
    return jnp.where(far, -1., 0.5)


def policy(obs):
    hand = obs[:3]
    mug = obs[4:7] + jnp.array([0.01, 0., 0.05])
    return action(move(hand, _desired_pos(hand, mug, obs[-3:-1]), p=10.), _grab_effort(hand, mug))
