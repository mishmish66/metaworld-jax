> **This is a slop fork of [Meta-World](https://github.com/Farama-Foundation/Metaworld).** It is an AI-written port
> of its manipulation tasks, scripted policies, and benchmarks to JAX on MuJoCo Warp, made by someone who did not
> write Meta-World. It is not affiliated with or endorsed by the Meta-World authors, and credit for the tasks, the
> robot models, and the scripted policies is theirs. The Meta-World parts stay under its MIT license
> (`THIRD_PARTY_NOTICES.md`); only what is new is under the VibeCoded AI-Slop License (`LICENSE`).

Meta-World's 50 Sawyer manipulation tasks as [jax_pomdps](https://github.com/mishmish66/jax-pomdps) POMDPs simulated
by MuJoCo Warp, with its scripted policies and benchmark task splits. Every method is a pure function of pytrees,
safe under `jax.jit` and `jax.vmap`, so thousands of worlds step together on a GPU inside a jitted training loop.

```python
import jax
import jax_pomdps as pomdps

env = pomdps.make("metaworld.env:pick-place", max_worlds=1024)
keys = jax.random.split(jax.random.key(0), 1024)
states = jax.jit(jax.vmap(env.reset))(keys)
states = jax.jit(jax.vmap(env.step))(keys, states, jax.vmap(env.action_space.sample)(keys))
obs = jax.jit(jax.vmap(env.observe))(keys, states, None)
```

- `metaworld.env`: the tasks, their observations, actions, rewards, and success conditions.
- `metaworld.policies`: Meta-World's scripted policies, as JAX functions of an observation.
- `metaworld.benchmarks`: the task splits of the MT and ML benchmarks, and goal sets for them.

## What differs from Meta-World

Resets, observations, rewards, info, and success follow Meta-World v3 (3.1.1) exactly, including its quirks, such as
reward terms that read the current finger pad positions where the code names initial ones. The tests check them
against episodes recorded from Meta-World itself: at recorded states, observations match to 1e-4 and rewards to
1e-3, except in the seven tasks whose hand reset is simulated (below). The scenes are Meta-World's own models. The
differences:

- **Simulator.** MuJoCo Warp simulates in float32 on the GPU, with MuJoCo 3.14, where Meta-World pins CPU MuJoCo
  3.3.0. Recorded actions reproduce recorded trajectories closely for a few steps; over an episode they drift
  apart, as any two simulators do. With Meta-World's physics settings, Meta-World's scripted policies succeed as
  often as in Meta-World (0.99 against 0.98 mean success over the 50 tasks).
- **Physics settings.** Each task simulates by default with the fastest preset it trains with (`Task.physics`). For
  most tasks that is `physics="fast"`: each step is 3 substeps of 4.17 ms, about 1.5× faster than Meta-World's 5
  substeps of 2.5 ms, which `physics="upstream"` keeps. The scripted policies succeed as often with either (0.98 and
  0.99 mean success over the 50 tasks, against Meta-World's 0.98); fewer, longer substeps break grasping.
  Coffee-push, shelf-place, stick-pull, and stick-push use `upstream`, where PPO (below) solves at least two more of
  8 seeds with it: 7, 6, 6, and 7 against 0, 4, 3, and 5 with `fast`. Passing `physics` overrides the task's preset.
  `benchmarks/README.md` has the measurements.
- **Hand reset.** Meta-World starts each episode by simulating 250 substeps that move the hand to its initial
  position. Here that state is recorded from Meta-World (`assets/hand_resets.npz`) and loaded, so resets cost no
  simulation. In seven tasks (box-close, disassemble, door-lock, door-unlock, handle-press, handle-pull,
  peg-unplug-side) the hand reset depends on where the task places its objects, so it is simulated each episode and
  ends slightly differently from Meta-World's; in door-lock and door-unlock the gripper presses on the lock during
  the reset, and the outcome differs by up to 2 cm.
- **Corrected rewards.** With `reward="corrected"`, tasks whose Meta-World reward pays for failing use a corrected
  one; the others, and success everywhere, stay Meta-World's. Assembly's placement reward pays most for holding the
  ring beside the peg below its top, where it cannot slide on; the corrected one pays most for centering the ring over
  the peg first. With PPO (below, without the entropy bonus) it solves 7 of 8 seeds against Meta-World's reward's 4.
- **Diverged worlds.** A world whose positions, velocities, or accelerations become non-finite or exceed 1e10 restarts
  from the model's initial state, as MuJoCo's `mj_step` resets diverged data; Meta-World inherits that from MuJoCo.
- **Contacts in info.** `grasp_success` tests whether both finger pads touch the object by distance, where
  Meta-World tests for contact forces; they agree on at least 95% of recorded steps.
- **Bin-picking.** Meta-World fixes one reward margin at the first reward computed; here it is fixed at reset,
  when the cube is at rest at the same place.
- **Images.** Camera images come from the MuJoCo Warp ray tracer, shaded to approximate MuJoCo's OpenGL renderer.
  Sites, such as goal markers, are drawn as spheres of their size and color.
- **Benchmarks.** Goal sets follow Meta-World's distributions but are drawn with JAX's random numbers, so a seed
  does not give Meta-World's goals for that seed.
- **Gymnasium.** There is no Gymnasium interface, no wrappers, and no episode time limit inside the POMDP;
  `MAX_PATH_LENGTH` is Meta-World's 500-step horizon.

## Training

PPO from rlx with these settings solves most tasks: 8 seeds of 20M steps, 128 worlds per seed, 512-step rollouts, 10
epochs of 32 minibatches, learning rates 3e-4, gradient norms clipped to 0.5, a 64×64 network, an entropy bonus of
0.01, and training rewards scaled by 0.1. Seeds reaching 80% success, of 8, on the harder tasks:

| task | seeds | task | seeds |
| --- | --- | --- | --- |
| basketball | 8 | coffee-push | 7 |
| hand-insert | 8 | stick-push | 7 |
| lever-pull | 8 | soccer | 7 |
| peg-insert-side | 8 | pick-out-of-hole | 6 |
| bin-picking | 7 | shelf-place | 6 |
| coffee-pull | 7 | stick-pull | 6 |

The remaining tasks solve on all or nearly all seeds with these or earlier settings, assembly with
`reward="corrected"`. Push-back and pick-place-wall solve no seed: their rewards pay little for approaching the object
from where the hand starts, so policies settle away from it. Box-close, disassemble, and peg-unplug-side were not run
with these settings; their simulated hand resets make them about ten times slower per step.

## Credits and licenses

- **Meta-World**, © 2019 Meta-World Team, MIT: the tasks, models, meshes, textures, rewards, scripted policies,
  benchmark splits, and the history this repository forks. Cite McLean et al., "Meta-World+: An Improved,
  Standardized, RL Benchmark", NeurIPS 2025 Datasets and Benchmarks Track, and Yu et al., "Meta-World: A Benchmark
  and Evaluation for Multi-Task and Meta Reinforcement Learning", CoRL 2019.
- **dm_control**, © DeepMind Technologies Limited, Apache 2.0: the reward utilities Meta-World adapted.
- **MuJoCo, MJX, and MuJoCo Warp**, © DeepMind Technologies Limited, Apache 2.0: dependencies.
- **Everything else** is under the VibeCoded AI-Slop License v1.0, © 2026 Misha Lvovsky.

`THIRD_PARTY_NOTICES.md` lists file by file what comes from where, with the full license texts.
