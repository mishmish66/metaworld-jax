"""Meta-World's scripted policy for push-wall-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, obj, goal):
    obj = obj + jnp.array([-0.005, 0., 0.])
    behind_wall = (-0.1 <= obj[0]) & (obj[0] <= 0.3) & (0.65 <= obj[1]) & (obj[1] <= 0.75)
    beside_wall = (((-0.15 < obj[0]) & (obj[0] < 0.05)) | ((0.15 < obj[0]) & (obj[0] < 0.35))) \
        & (0.695 <= obj[1]) & (obj[1] <= 0.755)
    return select(
        (jnp.linalg.norm(hand[:2] - obj[:2]) > 0.02, obj + jnp.array([0., 0., 0.2])),
        (jnp.abs(hand[2] - obj[2]) > 0.04, obj + jnp.array([0., 0., 0.03])),
        (behind_wall, hand + jnp.array([-1., 0., 0.])),
        (beside_wall, hand + jnp.array([0., 1., 0.])),
        default=goal,
    )


def _grab_effort(hand, obj):
    far = (jnp.linalg.norm(hand[:2] - obj[:2]) > 0.02) | (jnp.abs(hand[2] - obj[2]) > 0.1)
    return jnp.where(far, 0., 0.6)


def policy(obs):
    hand, obj, goal = obs[:3], obs[4:7], obs[-3:]
    return action(move(hand, _desired_pos(hand, obj, goal), p=10.), _grab_effort(hand, obj))
