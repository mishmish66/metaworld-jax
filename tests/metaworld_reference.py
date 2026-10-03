"""
Record upstream Meta-World's trajectories under its scripted policies, for the tests to compare against.

    uv run --no-project --python 3.13 --with packaging --with "metaworld @ git+https://github.com/Farama-Foundation/Metaworld@59fc34d7" \
        python tests/metaworld_reference.py

For each task this writes `tests/assets/reference/<task>.npz` with episodes of MT1 tasks: per step the simulator
state, action, observation, reward, and info, the state after the hand reset, the model's body positions, and
the env's numeric attributes after reset. `tests/assets/reference/success.json` holds the scripted policies' success
rates over all 50 MT1 tasks of each task, and `tests/assets/reference/images.npz` upstream's camera images at
steps of the first episode of a few tasks.
"""
import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "assets" / "reference"
SEED = 0
# (MT1 task index, policy action noise, step limit) of each recorded episode
EPISODES = [(0, 0.0, 500), (1, 0.0, 500), (2, 0.3, 150)]
# steps recorded after the first success
AFTER_SUCCESS = 10
# tasks, camera, image size, and steps of the first episode whose images are recorded
IMAGE_TASKS = ["reach", "pick-place", "door-open", "drawer-open", "button-press-topdown", "soccer"]
IMAGE_CAMERA, IMAGE_SIZE, IMAGE_STEPS = "corner2", 84, (0, 20, 40)
INFO_KEYS = ["success", "near_object", "grasp_success", "grasp_reward", "in_place_reward", "obj_to_target", "unscaled_reward"]
SKIPPED_ATTRIBUTES = {"curr_path_length", "num_resets", "_last_stable_obs", "_prev_obs", "frame_skip", "width", "height"}


def _make(name, task, **kwargs):
    import metaworld
    from metaworld.policies import ENV_POLICY_MAP

    benchmark = metaworld.MT1(name, seed=SEED)
    env = benchmark.train_classes[name](**kwargs)
    env.set_task(benchmark.train_tasks[task])
    return env, ENV_POLICY_MAP[name]()


def _sim_state(env):
    d = env.data
    return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy())


def _attributes(env):
    out = {}
    for k, v in vars(env).items():
        if k in SKIPPED_ATTRIBUTES:
            continue
        if isinstance(v, (bool, int, float, np.floating, np.integer, np.bool_)):
            out[k] = np.asarray(v, float)
        elif isinstance(v, np.ndarray) and v.dtype.kind in "fib" and 0 < v.size <= 20:
            out[k] = v.astype(float)
    return out


def _episode(name, task, noise, limit):
    env, policy = _make(name, task)
    hand = {}
    reset_hand = env._reset_hand

    def recording_reset_hand(*args, **kwargs):
        reset_hand(*args, **kwargs)
        hand.update(_sim_state(env))

    env._reset_hand = recording_reset_hand
    obs, _ = env.reset()
    rng = np.random.default_rng(task)
    record = {"rand_vec": np.asarray(env._last_rand_vec, float), "body_pos": env.model.body_pos.copy()}
    record |= {f"hand_{k}": v for k, v in hand.items()}
    record |= {f"attr_{k}": v for k, v in _attributes(env).items()}
    steps = {k: [v] for k, v in _sim_state(env).items()} | {"obs": [obs.copy()]}
    steps |= {"action": [], "reward": []} | {k: [] for k in INFO_KEYS}
    succeeded_at = None
    for t in range(limit):
        action = policy.get_action(obs) + noise * rng.standard_normal(4)
        action = np.clip(action, -1., 1.).astype(np.float32)
        obs, reward, _, _, info = env.step(action)
        for k, v in _sim_state(env).items():
            steps[k].append(v)
        steps["obs"].append(obs.copy())
        steps["action"].append(action)
        steps["reward"].append(reward)
        for k in INFO_KEYS:
            steps[k].append(float(info[k]))
        if info["success"] and succeeded_at is None:
            succeeded_at = t
        if succeeded_at is not None and t >= succeeded_at + AFTER_SUCCESS:
            break
    env.close()
    return record | {k: np.asarray(v) for k, v in steps.items()}


def record_task(name):
    out = {}
    for i, (task, noise, limit) in enumerate(EPISODES):
        out |= {f"ep{i}/{k}": v for k, v in _episode(name, task, noise, limit).items()}
    np.savez_compressed(OUT / f"{name.removesuffix('-v3')}.npz", **out)
    return name


def record_images(name):
    """Upstream's images at `IMAGE_STEPS` of the noise-free first recorded episode of `name`."""
    task, _, _ = EPISODES[0]
    env, policy = _make(
        name, task, render_mode="rgb_array", camera_name=IMAGE_CAMERA, width=IMAGE_SIZE, height=IMAGE_SIZE,
    )
    obs, _ = env.reset()
    images = {}
    for t in range(max(IMAGE_STEPS) + 1):
        if t in IMAGE_STEPS:
            images[t] = env.render().copy()
        obs, *_ = env.step(np.clip(policy.get_action(obs), -1., 1.).astype(np.float32))
    env.close()
    return name.removesuffix("-v3"), np.stack([images[t] for t in IMAGE_STEPS])


def success_rate(name):
    """Fraction of the 50 MT1 tasks the scripted policy solves within 500 steps, and the mean steps it takes."""
    solved, steps = 0, []
    for task in range(50):
        env, policy = _make(name, task)
        obs, _ = env.reset()
        for t in range(env.max_path_length):
            obs, _, _, _, info = env.step(np.clip(policy.get_action(obs), -1., 1.).astype(np.float32))
            if info["success"]:
                solved += 1
                steps.append(t + 1)
                break
        env.close()
    return name, {"success_rate": solved / 50, "mean_steps": float(np.mean(steps)) if steps else None}


def main():
    import metaworld

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=list(metaworld.MT1.ENV_NAMES))
    parser.add_argument("--no-success-rates", action="store_true")
    parser.add_argument("--images-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool() as pool:
        images = dict(pool.imap_unordered(record_images, [f"{t}-v3" for t in IMAGE_TASKS]))
        np.savez_compressed(OUT / "images.npz", steps=np.array(IMAGE_STEPS), camera=IMAGE_CAMERA, **images)
        print("wrote", OUT / "images.npz")
        if args.images_only:
            return
        for name in pool.imap_unordered(record_task, args.tasks):
            print("recorded", name, flush=True)
        if not args.no_success_rates:
            rates = dict(pool.imap_unordered(success_rate, args.tasks))
            path = OUT / "success.json"
            previous = json.loads(path.read_text()) if path.exists() else {}
            rates = previous | {k.removesuffix("-v3"): v for k, v in rates.items()}
            path.write_text(json.dumps(dict(sorted(rates.items())), indent=1))
            print("wrote", path)


if __name__ == "__main__":
    main()
