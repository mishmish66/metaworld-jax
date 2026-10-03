"""
Meta-World's tasks as jax_pomdps POMDPs simulated by MuJoCo Warp, registered as "<name>/<observation mode>", with
names as in Meta-World without "-v3" and the modes of `OBSERVATION_MODES`, and as "<name>" in the default mode:

    import jax_pomdps as pomdps
    env = pomdps.make("metaworld.env:pick-place")

.. include:: ../../../docs/tasks.md
"""
from metaworld.env.pomdp import (
    CONTROL_TIMESTEP, DEFAULT_OBSERVATION_MODE, MAX_PATH_LENGTH, OBSERVATION_MODES, PHYSICS, PIXELS, MetaworldPOMDP,
    State, load_model,
)
from metaworld.env.tasks import INFO_KEYS, TASKS, Task

__all__ = [
    "CONTROL_TIMESTEP", "DEFAULT_OBSERVATION_MODE", "INFO_KEYS", "MAX_PATH_LENGTH", "OBSERVATION_MODES", "PHYSICS",
    "PIXELS", "MetaworldPOMDP", "State", "TASKS", "Task", "load_model",
]
