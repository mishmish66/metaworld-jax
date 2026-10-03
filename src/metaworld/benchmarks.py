"""
Meta-World's benchmarks: the tasks each trains and tests on, and fixed sets of goals per task. Meta-World draws
`N_GOALS` random reset vectors per task for each benchmark; a POMDP made with them resets to one of them:

    import jax
    from metaworld import benchmarks
    from metaworld.env import MetaworldPOMDP

    goals = benchmarks.goals("reach", jax.random.key(0))
    env = MetaworldPOMDP("reach", rand_vecs=goals, goal_observable=benchmarks.BENCHMARKS["ML10"].goal_observable)

MT1 and ML1 are any single task, trained on with goal sets drawn with different keys for training and testing in
ML1. The goals follow Meta-World's distributions but not its numpy random sequences, so a seed does not give
Meta-World's goals for that seed.
"""
from typing import NamedTuple

import jax

from metaworld.env import TASKS, load_model

# random reset vectors Meta-World draws per task for a benchmark
N_GOALS = 50


class Benchmark(NamedTuple):
    train: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def goal_observable(self) -> bool:
        """Whether the benchmark observes goals: multi-task benchmarks do, meta-learning ones do not."""
        return not self.test


BENCHMARKS = {
    "MT10": Benchmark(
        train=(
            "reach", "push", "pick-place", "door-open", "drawer-open", "drawer-close",
            "button-press-topdown", "peg-insert-side", "window-open", "window-close",
        ),
        test=(),
    ),
    "MT25": Benchmark(
        train=(
            "reach", "push", "pick-place", "door-open", "drawer-open", "drawer-close",
            "button-press-topdown", "peg-insert-side", "window-open", "window-close", "coffee-pull",
            "pick-out-of-hole", "disassemble", "pick-place-wall", "basketball", "stick-pull",
            "button-press-wall", "faucet-open", "door-lock", "lever-pull", "sweep-into", "faucet-close",
            "coffee-button", "button-press-topdown-wall", "dial-turn",
        ),
        test=(),
    ),
    "MT50": Benchmark(
        train=(
            "assembly", "basketball", "bin-picking", "box-close", "button-press-topdown",
            "button-press-topdown-wall", "button-press", "button-press-wall", "coffee-button",
            "coffee-pull", "coffee-push", "dial-turn", "disassemble", "door-close", "door-lock",
            "door-open", "door-unlock", "hand-insert", "drawer-close", "drawer-open", "faucet-open",
            "faucet-close", "hammer", "handle-press-side", "handle-press", "handle-pull-side",
            "handle-pull", "lever-pull", "pick-place-wall", "pick-out-of-hole", "pick-place", "plate-slide",
            "plate-slide-side", "plate-slide-back", "plate-slide-back-side", "peg-insert-side",
            "peg-unplug-side", "soccer", "stick-push", "stick-pull", "push", "push-wall", "push-back",
            "reach", "reach-wall", "shelf-place", "sweep-into", "sweep", "window-open", "window-close",
        ),
        test=(),
    ),
    "ML10": Benchmark(
        train=(
            "reach", "push", "pick-place", "door-open", "drawer-close", "button-press-topdown",
            "peg-insert-side", "window-open", "sweep", "basketball",
        ),
        test=(
            "drawer-open", "door-close", "shelf-place", "sweep-into", "lever-pull",
        ),
    ),
    "ML25": Benchmark(
        train=(
            "reach", "push", "pick-place", "door-open", "drawer-open", "drawer-close",
            "button-press-topdown", "peg-insert-side", "window-open", "window-close", "coffee-pull",
            "pick-out-of-hole", "disassemble", "pick-place-wall", "basketball", "stick-pull",
            "button-press-wall", "faucet-open", "door-lock", "lever-pull", "sweep-into", "faucet-close",
            "coffee-button", "button-press-topdown-wall", "dial-turn",
        ),
        test=(
            "basketball", "door-close", "shelf-place", "sweep", "button-press",
        ),
    ),
    "ML45": Benchmark(
        train=(
            "assembly", "basketball", "button-press-topdown", "button-press-topdown-wall", "button-press",
            "button-press-wall", "coffee-button", "coffee-pull", "coffee-push", "dial-turn", "disassemble",
            "door-close", "door-open", "drawer-close", "drawer-open", "faucet-open", "faucet-close",
            "hammer", "handle-press-side", "handle-press", "handle-pull-side", "handle-pull", "lever-pull",
            "pick-place-wall", "pick-out-of-hole", "push-back", "pick-place", "plate-slide",
            "plate-slide-side", "plate-slide-back", "plate-slide-back-side", "peg-insert-side",
            "peg-unplug-side", "soccer", "stick-push", "stick-pull", "push-wall", "push", "reach-wall",
            "reach", "shelf-place", "sweep-into", "sweep", "window-open", "window-close",
        ),
        test=(
            "bin-picking", "box-close", "hand-insert", "door-lock", "door-unlock",
        ),
    ),
}


def goals(task: str, key, n: int = N_GOALS) -> jax.Array:
    """`n` random reset vectors of `task`, drawn as Meta-World draws them."""
    sample = TASKS[task](load_model(task)).sample
    return jax.vmap(sample)(jax.random.split(key, n))
