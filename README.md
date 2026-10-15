# Meta-World on MuJoCo Warp

> **A slop fork of [Meta-World](https://github.com/Farama-Foundation/Metaworld):** an AI-written port, written by an
> AI coding agent and not by hand, and not affiliated with or endorsed by Meta-World's authors. It claims no part of
> Meta-World: the tasks, robot models, rewards, scripted policies, and benchmarks are its authors' work, under its
> MIT license and copyright, unchanged ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Cite Meta-World, not
> this repository.

Meta-World's 50 Sawyer manipulation tasks as [jax_pomdps](https://github.com/mishmish66/jax-pomdps) POMDPs on MuJoCo
Warp, with its scripted policies and MT/ML benchmark splits, all in JAX.

**[Documentation](https://mishmish66.github.io/metaworld-jax/)**: the tasks, what differs from Meta-World, and the API.

```python
import jax, jax_pomdps as pomdps

env = pomdps.make("metaworld.env:pick-place", max_worlds=1024)
keys = jax.random.split(jax.random.key(0), 1024)
states = jax.jit(jax.vmap(env.reset))(keys)
states = jax.jit(jax.vmap(env.step))(keys, states, jax.vmap(env.action_space.sample)(keys))
```

## Setup

```sh
uv sync
uv run pytest tests
```

The tests compare against episodes recorded from upstream Meta-World (`tests/assets/reference`), which
`tests/metaworld_reference.py` records; `scripts/export_hand_resets.py` exports the recorded hand resets the tasks
start from.

[Benchmarks](benchmarks/README.md) compare throughput and fidelity with upstream Meta-World, and the physics settings.

## License

The Meta-World parts stay under its MIT license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Everything new
is under the [VibeCoded AI-Slop License v1.0](LICENSE), with no copyright claimed. If you use the tasks, cite McLean et al., "Meta-World+: An
Improved, Standardized, RL Benchmark", NeurIPS 2025, and Yu et al., "Meta-World: A Benchmark and Evaluation for
Multi-Task and Meta Reinforcement Learning", CoRL 2019.
