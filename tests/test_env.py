"""
Tests of the MuJoCo Warp POMDPs, with Meta-World's physics, against upstream Meta-World's recorded episodes in
assets/reference, for the comma-separated tasks $METAWORLD_TASKS (all by default).
"""
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax_pomdps as pomdps

import reference
from metaworld.env import DEFAULT_OBSERVATION_MODE, INFO_KEYS, OBSERVATION_MODES, TASKS, MetaworldPOMDP

TASK_NAMES = os.environ.get("METAWORLD_TASKS", ",".join(sorted(TASKS))).split(",")
NUM_WORLDS = 8
# policy steps of open-loop replay that track the recorded hand closely
REPLAY_STEPS = 10


@pytest.fixture(scope="module", params=TASK_NAMES)
def env(request):
    return MetaworldPOMDP(request.param, physics="upstream", max_worlds=NUM_WORLDS)


def _name(env):
    return next(name for name, cls in TASKS.items() if type(env.task) is cls)


def test_tasks_are_registered_pomdps(env):
    assert isinstance(env, pomdps.POMDP)
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (39,)


def test_hidden_goals_are_observed_as_zeros():
    env = MetaworldPOMDP("reach", goal_observable=False, max_worlds=1)
    state = jax.jit(env.reset)(jax.random.key(0))
    obs = jax.jit(env.observe)(None, state, jnp.zeros(4))
    np.testing.assert_array_equal(np.asarray(obs[-3:]), np.zeros(3))
    assert bool(jnp.any(state.episode["target_pos"] != 0.))


def test_tasks_register_in_each_observation_mode_and_alone_in_the_default_mode():
    names = {n for n in pomdps.registered() if n.split("/")[0] in TASKS}
    assert names == set(TASKS) | {f"{t}/{m}" for t in TASKS for m in OBSERVATION_MODES}
    default = pomdps.make("metaworld.env:reach", max_worlds=1)
    assert default.observation_space == pomdps.make(f"metaworld.env:reach/{DEFAULT_OBSERVATION_MODE}", max_worlds=1).observation_space


def test_unknown_tasks_are_rejected():
    with pytest.raises(ValueError):
        MetaworldPOMDP("reach-v3")


def test_resets_match_reference(env):
    for i in range(reference.EPISODES):
        ep = reference.episode(_name(env), i)
        state = jax.jit(env.init_state)(ep["rand_vec"])
        np.testing.assert_allclose(np.asarray(state.episode["target_pos"]), ep["obs"][0][-3:], atol=1e-4)
        np.testing.assert_allclose(np.asarray(env._current(state)), ep["obs"][0][:18], atol=1e-2, err_msg=f"episode {i}")


def test_observations_rewards_and_info_match_reference_at_recorded_states(env):
    for i in range(reference.EPISODES):
        ep = reference.episode(_name(env), i)
        steps = np.arange(1, len(ep["reward"]) + 1)
        states = jax.jit(lambda: reference.recorded_states(env, ep, steps))()
        actions = jnp.asarray(ep["action"])
        obs = jax.jit(jax.vmap(env.observe))(None, states, actions)
        reward, info = jax.jit(jax.vmap(env.evaluate))(states, actions)
        np.testing.assert_allclose(np.asarray(obs), ep["obs"][1:], atol=1e-4, err_msg=f"episode {i}")
        np.testing.assert_allclose(np.asarray(reward), ep["reward"], atol=1e-3, err_msg=f"episode {i}")
        for k in INFO_KEYS:
            if k == "grasp_success":
                # contact by distance stands in for upstream's contact force
                assert np.mean(np.asarray(info[k]) == ep[k]) >= 0.95, (i, k)
            else:
                np.testing.assert_allclose(np.asarray(info[k]), ep[k], atol=1e-3, err_msg=f"episode {i} {k}")


def test_open_loop_replay_tracks_reference_hand(env):
    ep = reference.episode(_name(env), 0)
    state = env.init_state(ep["rand_vec"])
    step = jax.jit(lambda s, a: env.step(None, s, a))
    for t in range(REPLAY_STEPS):
        state = step(state, jnp.asarray(ep["action"][t]))
    obs = jax.jit(env.observe)(None, state, None)
    np.testing.assert_allclose(np.asarray(obs[:3]), ep["obs"][REPLAY_STEPS][:3], atol=5e-3)


def test_batched_reset_step_observe_under_jit_and_vmap(env):
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    states = jax.jit(jax.vmap(env.reset))(keys)
    actions = jax.vmap(env.action_space.sample)(keys)
    next_states = jax.jit(jax.vmap(env.step))(keys, states, actions)
    obs = jax.jit(jax.vmap(env.observe))(keys, next_states, actions)
    rewards = jax.jit(jax.vmap(env.reward))(keys, states, actions, next_states)
    dones = jax.jit(jax.vmap(env.done))(next_states)
    assert not bool(jnp.any(jax.vmap(env.overflowed)(next_states)))
    assert obs.shape == (NUM_WORLDS, 39)
    assert rewards.shape == dones.shape == (NUM_WORLDS,)
    assert bool(jnp.all(jnp.isfinite(obs))) and bool(jnp.all(jnp.isfinite(rewards)))
    assert bool(jnp.all(jax.vmap(env.observation_space.contains)(obs)))


def test_sampled_resets_lie_in_the_reset_space(env):
    keys = jax.random.split(jax.random.key(1), 64)
    rand_vecs = np.asarray(jax.jit(jax.vmap(env.sample_rand_vec))(keys))
    low, high = np.asarray(env.task.rand_low), np.asarray(env.task.rand_high)
    assert np.all(rand_vecs >= low - 1e-6) and np.all(rand_vecs <= high + 1e-6)
    assert len(np.unique(rand_vecs.round(6), axis=0)) == len(keys)


IMAGES = np.load(reference.REFERENCE / "images.npz")
# mean absolute difference in 8-bit levels allowed between our and upstream's images
IMAGE_TOLERANCE = 12.


@pytest.mark.parametrize("task", [k for k in IMAGES.files if k in TASKS])
def test_camera_images_resemble_upstream_images(task):
    expected = IMAGES[task]
    steps = IMAGES["steps"]
    camera = str(IMAGES["camera"])
    env = MetaworldPOMDP(task, physics="upstream", camera_names=(camera,), camera_height=expected.shape[1], camera_width=expected.shape[2],
                         max_worlds=len(steps))
    states = jax.jit(lambda: reference.recorded_states(env, reference.episode(task, 0), steps))()
    images = np.asarray(jax.jit(jax.vmap(env.observe))(None, states, None)[f"{camera}_image"])
    assert images.dtype == np.uint8 and images.shape == expected.shape
    assert np.abs(images.astype(float) - expected).mean() < IMAGE_TOLERANCE


def test_pixel_modes_observe_images_and_proprioception():
    for mode, keys in {"pix": {"corner2_image"}, "pix-prp": {"corner2_image", "proprio"}}.items():
        env = pomdps.make(f"metaworld.env:reach/{mode}", max_worlds=1)
        assert set(env.observation_space.spaces) == keys, mode
        obs = jax.jit(env.observe)(None, jax.jit(env.reset)(jax.random.key(0)), None)
        assert set(obs) == keys, mode
        assert obs["corner2_image"].shape == (84, 84, 3) and obs["corner2_image"].dtype == jnp.uint8
        if "proprio" in keys:
            assert obs["proprio"].shape == env.observation_space.spaces["proprio"].shape == (8,)


@pytest.mark.parametrize("task", ["reach", "push", "pick-place"])
def test_images_show_the_goal(task):
    env = pomdps.make(f"metaworld.env:{task}/pix", max_worlds=1)
    state = jax.jit(env.reset)(jax.random.key(0))
    moved = state._replace(episode=state.episode | {"target_pos": state.episode["target_pos"] + jnp.array([0., 0.05, 0.05])})
    observe = jax.jit(env.observe)
    assert not np.array_equal(np.asarray(observe(None, state, None)["corner2_image"]),
                              np.asarray(observe(None, moved, None)["corner2_image"]))


def test_camera_observations_match_their_spaces_unbatched_and_mapped():
    env = MetaworldPOMDP("pick-place", camera_names=("corner", "gripperPOV"), camera_height=48, camera_width=64,
                         camera_depths=True, max_worlds=NUM_WORLDS)
    space = env.observation_space.spaces
    assert space["corner_image"].shape == (48, 64, 3) and space["gripperPOV_depth"].shape == (48, 64, 1)
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    mapped = jax.jit(jax.vmap(env.observe))(keys, jax.jit(jax.vmap(env.reset))(keys), None)
    single = jax.jit(env.observe)(keys[1], jax.jit(env.reset)(keys[1]), None)
    for k in ["state", "corner_image", "gripperPOV_image", "corner_depth", "gripperPOV_depth"]:
        assert mapped[k].shape == (NUM_WORLDS,) + space[k].shape, k
        np.testing.assert_allclose(np.asarray(single[k]), np.asarray(mapped[k][1]), atol=1e-5, err_msg=k)
    assert int(jnp.ptp(mapped["corner_image"])) > 0


def test_diverged_worlds_restart_from_the_initial_state():
    env = MetaworldPOMDP("peg-unplug-side", max_worlds=1)
    state = jax.jit(env.reset)(jax.random.key(0))
    diverged = state._replace(data=state.data.replace(qvel=state.data.qvel.at[9:].set(1e12)))
    after = jax.jit(lambda s: env.step(None, s, jnp.zeros(4)))(diverged)
    # the plug, at rest in the initial state, barely moves in the substeps after the restart
    np.testing.assert_allclose(np.asarray(after.data.qpos[9:]), env.sim.model.qpos0[9:], atol=1e-3)
    assert bool(jnp.all(jnp.isfinite(jax.jit(env.observe)(None, after, None))))


def test_observing_nothing_is_rejected():
    with pytest.raises(ValueError):
        MetaworldPOMDP("reach", observe_state=False)


@pytest.mark.parametrize("task", ["reach", "door-lock"])
def test_states_share_no_buffers_with_the_pomdp(task):
    env = MetaworldPOMDP(task, max_worlds=NUM_WORLDS)
    reset = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(jax.random.key(0), NUM_WORLDS)
    for x in jax.tree.leaves(reset(keys)):
        x.delete()
    assert bool(jnp.all(jnp.isfinite(reset(keys).data.qpos)))


def test_states_have_no_two_leaves_in_one_buffer(env):
    states = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2))
    buffers = [x.unsafe_buffer_pointer() for x in jax.tree.leaves(states) if x.size]
    assert len(buffers) == len(set(buffers))
