"""Meta-World's scripted policy for peg-insert-side-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, peg, goal):
    hole = jnp.array([-0.35, goal[1], 0.16])
    return select(
        (jnp.linalg.norm(hand[:2] - peg[:2]) > 0.04, peg + jnp.array([0., 0., 0.3])),
        (jnp.abs(hand[2] - peg[2]) > 0.025, peg),
        (jnp.linalg.norm(peg[1:] - hole[1:]) > 0.03, hole + jnp.array([0.4, 0., 0.])),
        default=hole,
    )


def _grab_effort(hand, peg):
    far = (jnp.linalg.norm(hand[:2] - peg[:2]) > 0.04) | (jnp.abs(hand[2] - peg[2]) > 0.15)
    return jnp.where(far, -1., 0.6)


def policy(obs):
    hand, peg, goal = obs[:3], obs[4:7], obs[-3:]
    return action(move(hand, _desired_pos(hand, peg, goal), p=25.), _grab_effort(hand, peg))
