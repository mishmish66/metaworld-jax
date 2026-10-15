"""
Export each task's state after upstream Meta-World's hand reset with the model's layout, to
src/metaworld/assets/hand_resets.npz, with the joint positions before its last simulator step, which the
kinematics in its data are computed from.

    uv run --no-project --python 3.13 --with packaging \
        --with "metaworld @ git+https://github.com/Farama-Foundation/Metaworld@59fc34d7" python scripts/export_hand_resets.py
"""
from pathlib import Path

import mujoco
import numpy as np

import metaworld

OUT = Path(__file__).parents[1] / "src" / "metaworld" / "assets" / "hand_resets.npz"
FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart")


def main():
    out = {}
    for name in metaworld.MT1.ENV_NAMES:
        env = metaworld.MT1(name, seed=0).train_classes[name]()
        m, d = env.model, env.data
        mujoco.mj_resetData(m, d)
        step = mujoco.mj_step
        last = {}

        def recording_step(m, d, nstep=1):
            for _ in range(nstep):
                last["qpos"] = d.qpos.copy()
                step(m, d)

        mujoco.mj_step = recording_step
        env._reset_hand()
        mujoco.mj_step = step
        task = name.removesuffix("-v3")
        out |= {f"{task}/{k}": np.array(getattr(d, k)) for k in FIELDS}
        out[f"{task}/kinematics_qpos"] = last["qpos"]
        env.close()
        print("exported", task, flush=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
