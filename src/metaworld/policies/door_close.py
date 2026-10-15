"""Meta-World's scripted policy for door-close-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, door, goal):
    door = door + jnp.array([0.05, 0.12, 0.1])
    beside = jnp.where(
        hand[2] < door[2] + 0.2,
        jnp.array([hand[0], hand[1], door[2] + 0.25]),
        jnp.array([door[0] - 0.02, door[1], hand[2]]),
    )
    return select(
        (hand[0] > door[0], beside),
        (jnp.abs(hand[2] - door[2]) > 0.04, door + jnp.array([-0.02, 0., 0.])),
        default=goal,
    )


def policy(obs):
    hand = obs[:3]
    return action(move(hand, _desired_pos(hand, obs[4:7], obs[-3:]), p=25.), 1.)
