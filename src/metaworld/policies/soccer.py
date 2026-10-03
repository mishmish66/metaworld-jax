"""Meta-World's scripted policy for soccer-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select

# distance behind the ball the hand pushes from
_OFFSET = 0.03


def _desired_pos(hand, ball, goal):
    ball = ball + jnp.array([0., 0., 0.03])
    desired_z = jnp.where(jnp.linalg.norm(hand[:2] - ball[:2]) < 0.02, 0.1, 0.03)
    push_location = select(
        (ball[0] - goal[0] < -0.05, ball + jnp.array([-_OFFSET, 0., 0.])),
        (ball[0] - goal[0] > 0.05, ball + jnp.array([_OFFSET, 0., 0.])),
        default=ball + jnp.array([0., -_OFFSET, 0.]),
    ).at[2].set(desired_z)
    return jnp.where(jnp.linalg.norm(hand - push_location) > 0.01, push_location, ball)


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7], obs[-3:]), p=25.), 1.)
