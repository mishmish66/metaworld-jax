"""Meta-World's scripted policy for pick-place-wall-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, puck, goal):
    puck = puck + jnp.array([-0.005, 0., 0.])
    over_wall = (-0.15 <= hand[0]) & (hand[0] <= 0.35) & (0.60 <= hand[1]) & (hand[1] <= 0.80)
    return select(
        (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.015, puck + jnp.array([0., 0., 0.1])),
        ((jnp.abs(hand[2] - puck[2]) > 0.04) & (puck[-1] < 0.03), puck + jnp.array([0., 0., 0.03])),
        # rise straight up while the wall is in the way
        (over_wall & (hand[2] < 0.25), hand + jnp.array([0., 0., 1.])),
        (over_wall & (hand[2] < 0.35), jnp.array([goal[0], goal[1], hand[2]])),
        (jnp.abs(hand[2] - goal[2]) > 0.01, jnp.array([hand[0], hand[1], goal[2]])),
        default=goal,
    )


def _grab_effort(hand, puck):
    far = (jnp.linalg.norm(hand[:2] - puck[:2]) > 0.015) | (jnp.abs(hand[2] - puck[2]) > 0.1)
    return jnp.where(far, 0., 0.9)


def policy(obs):
    hand, puck, goal = obs[:3], obs[4:7], obs[-3:]
    return action(move(hand, _desired_pos(hand, puck, goal), p=10.), _grab_effort(hand, puck))
