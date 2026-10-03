"""Meta-World's scripted policy for hammer-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select

_GOAL = jnp.array([0.24, 0.71, 0.11]) + jnp.array([-0.19, 0., 0.05])


def _desired_pos(hand, hammer):
    return select(
        (jnp.linalg.norm(hand[:2] - hammer[:2]) > 0.04, hammer + jnp.array([0., 0., 0.1])),
        ((jnp.abs(hand[2] - hammer[2]) > 0.05) & (hammer[-1] < 0.03), hammer + jnp.array([0., 0., 0.03])),
        (jnp.linalg.norm(hand[jnp.array([0, 2])] - _GOAL[jnp.array([0, 2])]) > 0.02, jnp.array([_GOAL[0], hand[1], _GOAL[2]])),
        default=_GOAL,
    )


def _grab_effort(hand, hammer):
    far = (jnp.linalg.norm(hand[:2] - hammer[:2]) > 0.04) | (jnp.abs(hand[2] - hammer[2]) > 0.1)
    return jnp.where(far, 0., 0.8)


def policy(obs):
    hand, hammer = obs[:3], obs[4:7] + jnp.array([-0.04, 0., -0.01])
    return action(move(hand, _desired_pos(hand, hammer), p=10.), _grab_effort(hand, hammer))
