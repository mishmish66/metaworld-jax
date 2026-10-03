"""Meta-World's scripted policy for disassemble-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, wrench):
    return select(
        (jnp.linalg.norm(hand[:2] - wrench[:2]) > 0.02, wrench + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - wrench[2]) > 0.03, wrench),
        default=hand + jnp.array([0., 0., 0.1]),
    )


def _grab_effort(hand, wrench):
    far = (jnp.linalg.norm(hand[:2] - wrench[:2]) > 0.02) | (jnp.abs(hand[2] - wrench[2]) > 0.07)
    return jnp.where(far, 0., 0.8)


def policy(obs):
    hand = obs[:3]
    wrench = obs[4:7] + jnp.array([-0.02, 0., 0.01])
    return action(move(hand, _desired_pos(hand, wrench), p=10.), _grab_effort(hand, wrench))
