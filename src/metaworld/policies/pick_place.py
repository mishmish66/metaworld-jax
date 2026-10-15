"""Meta-World's scripted policy for pick-place-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, gripper, puck, goal):
    puck = puck + jnp.array([-0.005, 0., 0.])
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.02, puck + jnp.array([0., 0., 0.1])),
        ((jnp.abs(hand[2] - puck[2]) > 0.05) & (puck[-1] < 0.04), puck + jnp.array([0., 0., 0.03])),
        # wait for the gripper to close
        (gripper > 0.73, hand),
        default=goal,
    )


def policy(obs):
    hand, gripper, puck, goal = obs[:3], obs[3], obs[4:7], obs[-3:]
    grab = jnp.where(jnp.linalg.norm(hand - puck) < 0.07, 1., 0.)
    return action(move(hand, _desired_pos(hand, gripper, puck, goal), p=10.), grab)
