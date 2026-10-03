# Third-party notices

This repository is a slop fork: an AI-written port of Meta-World's manipulation tasks to JAX on MuJoCo Warp.
This file lists, file by file, where each part comes from, who made it, and under which license, followed by
the full license texts. The repository's own `LICENSE` covers only what is new; everything below stays under its
original license.

## Where each part comes from

### Meta-World (MIT)

Meta-World, © 2019 Meta-World Team, MIT license (full text below). <https://github.com/Farama-Foundation/Metaworld>.
This repository's history starts as a fork of Meta-World 3.1.1 (`59fc34d7`), and the port follows its v3 tasks.

| file | from Meta-World |
|---|---|
| `src/metaworld/assets/` | the task models, meshes, and textures, copied |
| `src/metaworld/assets/hand_resets.npz` | the state after each task's hand reset, exported from Meta-World's environments by `scripts/export_hand_resets.py` |
| `src/metaworld/env/pomdp.py` | the mocap control of the hand and gripper, the observation layout and bounds, and the order of a step (`sawyer_xyz_env.py`) |
| `src/metaworld/env/tasks/*.py` | the resets, observations, rewards, and success checks of `envs/sawyer_*_v3.py` and `SawyerXYZEnv._gripper_caging_reward` |
| `src/metaworld/env/rewards.py` | `utils/reward_utils.py`, which credits the authors of dm_control |
| `src/metaworld/policies/*.py` | the scripted policies of `policies/sawyer_*_v3_policy.py` |
| `tests/metaworld_reference.py`, `scripts/export_hand_resets.py` | run Meta-World itself, installed when the script runs |
| `tests/assets/reference/` | episodes recorded from Meta-World by `tests/metaworld_reference.py` |

Resets, observations, rewards, success checks, and policies in these files are Meta-World's, re-expressed in JAX.

### dm_control (Apache 2.0)

dm_control, © DeepMind Technologies Limited, Apache License 2.0. <https://github.com/google-deepmind/dm_control>.
Meta-World's `reward_utils.py`, ported in `src/metaworld/env/rewards.py`, follows dm_control's `rewards.tolerance`.

### MuJoCo, MJX, and MuJoCo Warp (Apache 2.0)

MuJoCo, MJX, and MuJoCo Warp, © DeepMind Technologies Limited, Apache License 2.0.
<https://github.com/google-deepmind/mujoco>, <https://github.com/google-deepmind/mujoco_warp>. They are
dependencies, not copied. `src/metaworld/env/geometry.py` calls MJX's convex collision functions.

## License texts

### Meta-World

```
MIT License

Copyright (c) 2019 Meta-World Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
