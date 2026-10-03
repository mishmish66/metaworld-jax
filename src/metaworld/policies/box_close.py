"""Meta-World's scripted policy for box-close-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, lid, box):
    return select(
        (jnp.linalg.norm(hand[:2] - lid[:2]) > 0.01, jnp.array([lid[0], lid[1], 0.2])),
        (jnp.abs(hand[2] - lid[2]) > 0.05, lid),
        (jnp.abs(hand[2] - box[2]) > 0.04, jnp.array([hand[0], hand[1], box[2]])),
        default=box,
    )


def _grab_effort(hand, lid):
    far = (jnp.linalg.norm(hand[:2] - lid[:2]) > 0.01) | (jnp.abs(hand[2] - lid[2]) > 0.13)
    return jnp.where(far, 0.5, 1.)


def policy(obs):
    hand = obs[:3]
    lid = obs[4:7] + jnp.array([0., 0., 0.02])
    box = jnp.array([obs[-3], obs[-2], 0.15])
    return action(move(hand, _desired_pos(hand, lid, box), p=25.), _grab_effort(hand, lid))
