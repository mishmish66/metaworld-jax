"""Meta-World's scripted policy for stick-pull-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select

# grasp point on the stick relative to its observed position
_STICK_OFFSET = jnp.array([-0.015, 0., 0.03])


def _desired_pos(hand, stick, thermos, goal):
    stick = stick + _STICK_OFFSET
    thermos = thermos + jnp.array([-0.015, 0., 0.03])
    goal = goal + jnp.array([-0.05, 0., 0.])
    to_thermos = select(
        (jnp.linalg.norm(hand[:2] - stick[:2]) > 0.02, stick + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - stick[2]) > 0.02, stick),
        (jnp.abs(stick[1] - thermos[1]) > 0.02, jnp.stack([stick[0], thermos[1], stick[2]])),
        (jnp.abs(stick[2] - thermos[2]) > 0.02, jnp.stack([stick[0], thermos[1], thermos[2]])),
        default=thermos,
    )
    return jnp.where(jnp.abs(stick[0] - thermos[0]) > 0.04, to_thermos, goal)


def _grab_effort(hand, stick):
    stick = stick + _STICK_OFFSET
    far = (jnp.linalg.norm(hand[:2] - stick[:2]) > 0.02) | (jnp.abs(hand[2] - stick[2]) > 0.1)
    return jnp.where(far, -1., 0.7)


def policy(obs):
    hand, stick, thermos, goal = obs[:3], obs[4:7], obs[11:14], obs[-3:]
    return action(move(hand, _desired_pos(hand, stick, thermos, goal), p=25.), _grab_effort(hand, stick))
