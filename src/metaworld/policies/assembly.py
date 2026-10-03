"""Meta-World's scripted policy for assembly-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, wrench, peg):
    return select(
        (jnp.linalg.norm(hand[:2] - wrench[:2]) > 0.02, wrench + jnp.array([0., 0., 0.1])),
        (jnp.linalg.norm(hand[:2] - peg[:2]) <= 0.02, peg + jnp.array([0., 0., -0.2])),
        (jnp.abs(hand[2] - wrench[2]) > 0.05, wrench + jnp.array([0., 0., 0.03])),
        (jnp.abs(hand[2] - peg[2]) > 0.04, jnp.array([hand[0], hand[1], peg[2]])),
        default=peg,
    )


def _grab_effort(hand, wrench):
    far = (jnp.linalg.norm(hand[:2] - wrench[:2]) > 0.02) | (jnp.abs(hand[2] - wrench[2]) > 0.12)
    return jnp.where(far, 0., 0.6)


def policy(obs):
    hand = obs[:3]
    wrench = obs[4:7] + jnp.array([-0.02, 0., 0.])
    peg = obs[-3:] + jnp.array([0.12, 0., 0.14])
    return action(move(hand, _desired_pos(hand, wrench, peg), p=10.), _grab_effort(hand, wrench))
