"""Meta-World's scripted policy for peg-unplug-side-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, peg):
    return select(
        (jnp.linalg.norm(hand[:2] - peg[:2]) > 0.04, peg + jnp.array([0., 0., 0.2])),
        (jnp.abs(hand[2] - 0.15) > 0.02, jnp.array([peg[0], peg[1], 0.15])),
        default=hand + jnp.array([0.01, 0., 0.]),
    )


def _grab_effort(hand, peg):
    far = (jnp.linalg.norm(hand[:2] - peg[:2]) > 0.04) | (jnp.abs(hand[2] - peg[2]) > 0.15)
    return jnp.where(far, -1., 0.1)


def policy(obs):
    hand, peg = obs[:3], obs[4:7] + jnp.array([-0.02, 0., 0.035])
    return action(move(hand, _desired_pos(hand, peg), p=25.), _grab_effort(hand, peg))
