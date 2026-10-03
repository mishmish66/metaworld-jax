"""Meta-World's scripted policy for faucet-open-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, faucet):
    faucet = faucet + jnp.array([-0.04, 0., 0.03])
    return select(
        (jnp.linalg.norm(hand[:2] - faucet[:2]) > 0.04, faucet + jnp.array([0., 0., 0.1])),
        (jnp.abs(hand[2] - faucet[2]) > 0.04, faucet),
        default=faucet + jnp.array([0.1, 0.05, 0.]),
    )


def policy(obs):
    hand, faucet = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, faucet), p=25.), 1.)
