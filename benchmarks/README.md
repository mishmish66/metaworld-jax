# Benchmarks: upstream Meta-World vs MuJoCo Warp

Upstream Meta-World simulates one environment per process on CPU MuJoCo. This repository simulates many worlds at once
on a GPU with MuJoCo Warp. The benchmarks compare them on:

- **Throughput**: policy steps per second under random actions. Upstream is timed in one process and in one process
  per CPU thread; MuJoCo Warp with 4096 worlds stepped together.
- **Fidelity**: the success rate of Meta-World's scripted policies over 50 random resets per task, and their mean
  steps to success, next to upstream's over the 50 goals of its MT1 benchmark (`tests/assets/reference/success.json`).
- **Physics settings**: the same measures under simplified simulation options, to choose how much physics to trade
  for speed.

```sh
benchmarks/run_physics.sh all upstream fast                    # this repo, one process per task
uv run benchmarks/upstream_baseline.py --tasks reach push ...  # upstream Meta-World, in its own environment
uv run python benchmarks/report.py                             # the tables below, from benchmarks/results/
```

`upstream_baseline.py` declares its own dependencies (PEP 723), so uv runs it with upstream Meta-World and MuJoCo
3.3.0 in a separate environment.

The `fast` preset, the default, is the `dt4ms` setting below. The GPU timings are noisy: the GPUs ran other jobs
during the measurements.

## Results

<!-- results -->

### Physics settings

Tasks: assembly, basketball, door-open, drawer-open, hammer, peg-insert-side, pick-place, push. Upstream Meta-World's mean success is 0.96. Speed is relative to `upstream` on the same GPU in the same run, so it is given only for the settings run alongside it.

| setting | MuJoCo options | mean success | worst task vs upstream | speed |
| --- | --- | --- | --- | --- |
| `iter5` | `{"iterations": 5, "ls_iterations": 8}` | 0.98 | +0.00 | — |
| `dt4ms` | `{"timestep": 0.004166666666666667}` | 0.96 | -0.02 | 1.53× |
| `dt6ms` | `{"timestep": 0.00625}` | 0.62 | -1.00 | — |
| `dt6ms_implicitfast` | `{"timestep": 0.00625, "integrator": 3}` | 0.62 | -1.00 | — |
| `dt6ms_iter5` | `{"timestep": 0.00625, "iterations": 5, "ls_iterations": 8}` | 0.62 | -1.00 | — |
| `upstream` | `{}` | 0.98 | +0.00 | 1.00× |

### Per task

The settings ran at different times on GPUs with different loads, so their speeds are compared in the table above, where they ran together.

| task | success (upstream) | success, `fast` | success, `upstream` | steps (upstream) | steps, `fast` | steps, `upstream` | steps/s, `upstream` | upstream steps/s, 1 process | upstream steps/s, all processes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assembly | 1.00 | 1.00 | 1.00 | 89 | 89 | 89 | 256k | 5k | 119k |
| basketball | 0.92 | 0.90 | 0.96 | 96 | 93 | 92 | 331k | 6k | 136k |
| bin-picking | 1.00 | 1.00 | 1.00 | 124 | 185 | 196 | 330k | — | — |
| box-close | 0.86 | 0.88 | 0.88 | 107 | 135 | 110 | 173k | — | — |
| button-press | 1.00 | 1.00 | 1.00 | 59 | 59 | 59 | 364k | — | — |
| button-press-topdown | 1.00 | 1.00 | 1.00 | 65 | 66 | 66 | 446k | — | — |
| button-press-topdown-wall | 1.00 | 1.00 | 1.00 | 57 | 57 | 57 | 441k | — | — |
| button-press-wall | 0.96 | 0.98 | 0.98 | 65 | 64 | 64 | 330k | — | — |
| coffee-button | 1.00 | 1.00 | 1.00 | 35 | 35 | 35 | 266k | — | — |
| coffee-pull | 1.00 | 1.00 | 1.00 | 79 | 80 | 79 | 344k | — | — |
| coffee-push | 1.00 | 1.00 | 1.00 | 54 | 54 | 55 | 325k | — | — |
| dial-turn | 1.00 | 1.00 | 1.00 | 72 | 79 | 74 | 412k | — | — |
| disassemble | 0.98 | 0.96 | 0.94 | 129 | 170 | 186 | 282k | — | — |
| door-close | 1.00 | 1.00 | 1.00 | 66 | 69 | 67 | 372k | — | — |
| door-lock | 1.00 | 1.00 | 1.00 | 84 | 107 | 117 | 288k | — | — |
| door-open | 0.92 | 0.92 | 0.94 | 82 | 82 | 80 | 258k | 5k | 131k |
| door-unlock | 1.00 | 1.00 | 1.00 | 45 | 48 | 44 | 285k | — | — |
| drawer-close | 1.00 | 1.00 | 1.00 | 78 | 78 | 79 | 313k | — | — |
| drawer-open | 1.00 | 1.00 | 1.00 | 89 | 88 | 88 | 330k | 7k | 196k |
| faucet-close | 1.00 | 1.00 | 1.00 | 65 | 65 | 65 | 405k | — | — |
| faucet-open | 1.00 | 1.00 | 1.00 | 59 | 62 | 59 | 401k | — | — |
| hammer | 1.00 | 1.00 | 1.00 | 67 | 65 | 66 | 251k | 5k | 113k |
| hand-insert | 1.00 | 1.00 | 1.00 | 60 | 59 | 59 | 320k | — | — |
| handle-press | 1.00 | 1.00 | 1.00 | 32 | 34 | 28 | 187k | — | — |
| handle-press-side | 1.00 | 1.00 | 1.00 | 41 | 71 | 42 | 184k | — | — |
| handle-pull | 1.00 | 1.00 | 1.00 | 106 | 169 | 103 | 258k | — | — |
| handle-pull-side | 1.00 | 1.00 | 1.00 | 80 | 110 | 77 | 263k | — | — |
| lever-pull | 1.00 | 0.94 | 1.00 | 79 | 94 | 80 | 294k | — | — |
| peg-insert-side | 0.82 | 0.88 | 0.92 | 102 | 101 | 97 | 339k | 4k | 101k |
| peg-unplug-side | 0.98 | 0.98 | 1.00 | 107 | 100 | 96 | 196k | — | — |
| pick-out-of-hole | 1.00 | 1.00 | 1.00 | 116 | 194 | 118 | 384k | — | — |
| pick-place | 1.00 | 1.00 | 1.00 | 53 | 53 | 53 | 410k | 2k | 65k |
| pick-place-wall | 1.00 | 1.00 | 1.00 | 85 | 90 | 85 | 324k | — | — |
| plate-slide | 1.00 | 1.00 | 1.00 | 51 | 50 | 51 | 345k | — | — |
| plate-slide-back | 1.00 | 1.00 | 1.00 | 64 | 69 | 64 | 348k | — | — |
| plate-slide-back-side | 1.00 | 1.00 | 1.00 | 45 | 46 | 45 | 338k | — | — |
| plate-slide-side | 1.00 | 1.00 | 1.00 | 48 | 49 | 48 | 380k | — | — |
| push | 1.00 | 1.00 | 1.00 | 61 | 61 | 61 | 404k | 2k | 65k |
| push-back | 0.94 | 0.98 | 1.00 | 159 | 210 | 193 | 460k | — | — |
| push-wall | 1.00 | 1.00 | 1.00 | 80 | 80 | 81 | 317k | — | — |
| reach | 1.00 | 1.00 | 1.00 | 48 | 48 | 48 | 410k | 5k | 137k |
| reach-wall | 1.00 | 1.00 | 1.00 | 47 | 47 | 46 | 307k | — | — |
| shelf-place | 0.94 | 0.94 | 1.00 | 99 | 101 | 89 | 306k | — | — |
| soccer | 0.88 | 0.82 | 0.86 | 80 | 81 | 81 | 335k | — | — |
| stick-pull | 0.90 | 0.82 | 0.92 | 129 | 194 | 126 | 241k | — | — |
| stick-push | 1.00 | 1.00 | 1.00 | 74 | 77 | 74 | 243k | — | — |
| sweep | 1.00 | 1.00 | 1.00 | 86 | 93 | 88 | 389k | — | — |
| sweep-into | 0.90 | 0.88 | 0.88 | 56 | 54 | 55 | 310k | — | — |
| window-close | 1.00 | 1.00 | 1.00 | 80 | 80 | 80 | 403k | — | — |
| window-open | 1.00 | 1.00 | 1.00 | 87 | 87 | 87 | 356k | — | — |
