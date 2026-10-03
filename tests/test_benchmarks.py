import numpy as np

import jax

from metaworld import benchmarks
from metaworld.env import TASKS, MetaworldPOMDP


def test_benchmarks_name_registered_tasks_and_observe_goals_only_when_multi_task():
    for name, benchmark in benchmarks.BENCHMARKS.items():
        assert set(benchmark.train + benchmark.test) <= set(TASKS), name
        assert benchmark.goal_observable == name.startswith("MT"), name
    assert len(benchmarks.BENCHMARKS["MT50"].train) == len(TASKS)


def test_goals_are_distinct_and_lie_in_the_reset_space():
    goals = np.asarray(benchmarks.goals("pick-place", jax.random.key(0)))
    task = TASKS["pick-place"]
    assert goals.shape == (benchmarks.N_GOALS, len(task.rand_low))
    assert np.all(goals >= np.asarray(task.rand_low) - 1e-6) and np.all(goals <= np.asarray(task.rand_high) + 1e-6)
    assert len(np.unique(goals, axis=0)) == benchmarks.N_GOALS


def test_resets_draw_from_the_given_goals():
    goals = benchmarks.goals("reach", jax.random.key(0), n=3)
    env = MetaworldPOMDP("reach", rand_vecs=goals, max_worlds=16)
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(1), 16))
    targets = np.asarray(states.episode["target_pos"])
    distance = np.abs(targets[:, None] - np.asarray(goals)[None, :, 3:]).max(-1).min(-1)
    assert np.all(distance < 1e-6)
    assert len(np.unique(targets.round(6), axis=0)) == 3
