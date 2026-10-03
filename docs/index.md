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
- **Physics settings.** By default (`physics="fast"`) each step is 3 substeps of 4.17 ms, about 1.5× faster than
  Meta-World's 5 substeps of 2.5 ms, which `physics="upstream"` keeps. The scripted policies succeed as often with
  either (0.98 and 0.99 mean success over the 50 tasks, against Meta-World's 0.98); fewer, longer substeps break
  grasping. `benchmarks/README.md` has the measurements.
- **Hand reset.** Meta-World starts each episode by simulating 250 substeps that move the hand to its initial
  position. Here that state is recorded from Meta-World (`assets/hand_resets.npz`) and loaded, so resets cost no
  simulation. In seven tasks (box-close, disassemble, door-lock, door-unlock, handle-press, handle-pull,
  peg-unplug-side) the hand reset depends on where the task places its objects, so it is simulated each episode and
  ends slightly differently from Meta-World's; in door-lock and door-unlock the gripper presses on the lock during
  the reset, and the outcome differs by up to 2 cm.
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

## Credits and licenses

- **Meta-World**, © 2019 Meta-World Team, MIT: the tasks, models, meshes, textures, rewards, scripted policies,
  benchmark splits, and the history this repository forks. Cite McLean et al., "Meta-World+: An Improved,
  Standardized, RL Benchmark", NeurIPS 2025 Datasets and Benchmarks Track, and Yu et al., "Meta-World: A Benchmark
  and Evaluation for Multi-Task and Meta Reinforcement Learning", CoRL 2019.
- **dm_control**, © DeepMind Technologies Limited, Apache 2.0: the reward utilities Meta-World adapted.
- **MuJoCo, MJX, and MuJoCo Warp**, © DeepMind Technologies Limited, Apache 2.0: dependencies.
- **Everything else** is under the VibeCoded AI-Slop License v1.0, © 2026 Misha Lvovsky.

`THIRD_PARTY_NOTICES.md` lists file by file what comes from where, with the full license texts.
