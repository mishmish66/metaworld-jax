"""Meta-World's scripted policy for reach-v3."""
from metaworld.policies.base import action, move


def policy(obs):
    return action(move(obs[:3], obs[-3:], p=5.), 0.)
