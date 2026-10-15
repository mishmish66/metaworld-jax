"""Meta-World's scripted policy for basketball-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, ball, hoop_x):
    ball = ball + jnp.array([0., 0., 0.01])
    hoop = jnp.array([hoop_x, 0.875, 0.35])
    return select(
        (jnp.linalg.norm(hand[:2] - ball[:2]) > 0.04, ball + jnp.array([0., 0., 0.3])),
        (jnp.abs(hand[2] - ball[2]) > 0.025, ball),
        (jnp.abs(ball[2] - hoop[2]) > 0.025, jnp.array([hand[0], hand[1], hoop[2]])),
        default=hoop,
    )


def _grab_effort(hand, ball):
    far = (jnp.linalg.norm(hand[:2] - ball[:2]) > 0.04) | (jnp.abs(hand[2] - ball[2]) > 0.15)
    return jnp.where(far, -1., 0.6)


def policy(obs):
    hand, ball = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, ball, obs[-3]), p=25.), _grab_effort(hand, ball))
