"""
Meta-World's tasks as jax_pomdps POMDPs simulated by MuJoCo Warp through MJX:

    import jax_pomdps as pomdps
import warp as wp
    env = pomdps.make("metaworld.env:pick-place")

Each task's model is Meta-World's; its arm follows a mocap body welded to the hand, as in Meta-World. Resets,
observations, rewards, and success are Meta-World's, re-expressed in JAX. All methods are pure functions of
pytrees, safe under `jax.jit` and `jax.vmap`.
"""
import functools
import os
from copy import deepcopy
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx.third_party.mujoco_warp import OverflowType

import jax_pomdps as pomdps
import warp as wp

from metaworld.env.render import CameraRenderer
from metaworld.env.tasks import TASKS, Task

_ASSETS = Path(__file__).parents[1] / "assets"

# seconds per policy step
CONTROL_TIMESTEP = 0.0125
# overrides of the models' simulation options: 3 substeps per policy step, and Meta-World's 5
PHYSICS = {
    "fast": dict(timestep=CONTROL_TIMESTEP / 3),
    "upstream": {},
}
# policy steps in a Meta-World episode
MAX_PATH_LENGTH = 500
# hand displacement in meters of a unit action
_ACTION_SCALE = 0.01
# orientation the mocap body holds the hand in
_MOCAP_QUAT = np.array([1., 0., 1., 0.]) / np.sqrt(2.)
# policy steps that move the hand to its initial position at reset
_HAND_RESET_STEPS = 50
# gripper controls while the hand resets
_HAND_RESET_CTRL = np.array([-1., 1.])
# position-dependent fields of world data
_KINEMATICS = ("xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat")
# fields of the recorded hand resets in assets/hand_resets.npz
_HAND_RESET_FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart")
# MuJoCo's mjMAXVAL: a world whose positions, velocities, or accelerations exceed it has diverged
_MAX_VALUE = 1e10
# relative pose and torque scale of the welds to mocap bodies
_MOCAP_WELD = np.array([0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 5.])
# bounds of the observed hand position
_HAND_LOW, _HAND_HIGH = np.array([-0.525, 0.348, -0.0525]), np.array([0.525, 1.025, 0.7])
# padded length of the observed object positions and quaternions
_OBJECT_OBS = 14
# maximum distance in meters between the fingers, which the observed gripper opening is a fraction of
_GRIPPER_SPAN = 0.1
# body of the spheres that show sites in images
_SITE_MARKERS = "site_markers"
# default (contacts, constraint rows) allocated per world, about twice the peaks of noisy scripted-policy episodes
_BUFFER_SIZES = (64, 512)
# share of the contact buffer usable by contacts that need convex collision (EPA), whose scratch memory each step
# allocates in proportion
_CCD_SHARE = 0.25
# overflows of fixed-size MuJoCo Warp buffers, which drop contacts or constraints
_BUFFER_OVERFLOWS = int(
    OverflowType.NEFC | OverflowType.NJMAX_NNZ | OverflowType.BROADPHASE
    | OverflowType.NARROWPHASE | OverflowType.CCD | OverflowType.EPA_HORIZON
)


def _keep_external_streams_registered():
    """
    Keep CUDA streams that Warp does not own (JAX's) registered with Warp for the life of the process. Warp wraps
    JAX's stream anew on every FFI call, and freeing any one wrapper must not unregister the stream while another
    wrapper is using it.
    """
    release = wp.Stream.__del__

    def __del__(self):
        if getattr(self, "owner", False):
            release(self)

    wp.Stream.__del__ = __del__


_keep_external_streams_registered()


def _visible_sites(m) -> np.ndarray:
    """Sites MuJoCo's renderer shows by default, as Meta-World renders them."""
    return np.flatnonzero((m.site_group <= 2) & (m.site_rgba[:, 3] > 0))


def load_model(task: str) -> mujoco.MjModel:
    """
    Meta-World's MuJoCo model of `task`, with its mocap welds and collision settings, and a non-colliding sphere for
    each visible site on a last, static body, which images show the sites with.
    """
    path = _ASSETS / "sawyer_xyz" / TASKS[task].xml
    plain = mujoco.MjModel.from_xml_path(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    markers = spec.worldbody.add_body(name=_SITE_MARKERS)
    for i in _visible_sites(plain):
        markers.add_geom(
            name=f"{_SITE_MARKERS}_{plain.site(i).name}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[plain.site_size[i][0], 0., 0.], rgba=plain.site_rgba[i], group=int(plain.site_group[i]),
            contype=0, conaffinity=0,
        )
    m = spec.compile()
    for i in np.flatnonzero(m.eq_type == mujoco.mjtEq.mjEQ_WELD):
        m.eq_data[i] = _MOCAP_WELD
    # MuJoCo's collision defaults of Meta-World's pinned version
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    return m


def _simulated(m, physics):
    """Copy of `m` as MuJoCo Warp simulates it, with simulation options `physics`."""
    m = deepcopy(m)
    for k, v in physics.items():
        setattr(m.opt, k, v)
    # box-box contacts with margins need MuJoCo Warp's primitive collider
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    return m


class State(NamedTuple):
    data: mjx.Data
    # unclipped object and hand observation of the previous state
    prev: jax.Array
    # positions of the bodies the task moves at reset
    layout: jax.Array
    # constants of the episode, as the task defines them
    episode: dict[str, Any]


class Sim:
    """Simulation of a task's model with the bodies it moves at reset placed per world."""
    def __init__(self, m, task: Task, physics, naconmax, njmax, graph_mode):
        self.model = m
        self.simulated = simulated = _simulated(m, physics)
        # simulator steps per policy step
        self.substeps = round(CONTROL_TIMESTEP / simulated.opt.timestep)
        if not np.isclose(self.substeps * simulated.opt.timestep, CONTROL_TIMESTEP):
            raise ValueError(f"the timestep must divide {CONTROL_TIMESTEP} s, not {simulated.opt.timestep}")
        self._moved = np.array([m.body(b).id for b in task.moved_bodies], dtype=int)
        self.default_layout = np.asarray(m.body_pos[self._moved], np.float32)
        # moved bodies and the bodies below them, and their geoms welded to the world
        below = np.isin(np.arange(m.nbody), self._moved)
        for b in range(1, m.nbody):
            below[b] |= below[m.body_parentid[b]]
        self._moved_bodies = np.flatnonzero(below)
        self._static_geoms = np.flatnonzero(
            (m.body_weldid[m.geom_bodyid] == 0) & np.isin(m.geom_bodyid, self._moved_bodies)
        )
        graph_mode = getattr(wp.JaxCallableGraphMode, graph_mode.upper())
        # overflows are reported per world by `overflowed`, not printed
        self._mx = mjx.put_model(simulated, impl="warp", graph_mode=graph_mode).tree_replace({"opt._impl.warn_overflow": 0})
        self._data_kwargs = dict(impl="warp", naconmax=naconmax, naccdmax=int(naconmax * _CCD_SHARE), njmax=njmax)

    def _placed(self, layout):
        if not len(self._moved):
            return self._mx
        return self._mx.tree_replace({"body_pos": self._mx.body_pos.at[self._moved].set(layout)})

    def make_data(self, layout) -> mjx.Data:
        """World data at the model's initial state with the moved bodies at `layout`."""
        return self.place(mjx.make_data(self.model, **self._data_kwargs), layout)

    def put_data(self, d: mujoco.MjData) -> mjx.Data:
        """World data of MuJoCo data `d`, with its kinematics, and with the moved bodies at their model positions."""
        kinematics = {k: getattr(d, k).copy() for k in _KINEMATICS}
        # putting data recomputes the kinematics of `d`
        out = mjx.put_data(self.model, d, **self._data_kwargs)
        return out.replace(**{
            k: jnp.asarray(v, getattr(out, k).dtype).reshape(getattr(out, k).shape) for k, v in kinematics.items()
        })

    def place(self, d, layout):
        """`d` with the kinematics of the bodies below moved bodies, and of their geoms and sites, at `layout`."""
        if not len(self._moved):
            return d
        placed = self.kinematics(d, layout)
        m = self.model
        body = self._moved_bodies
        geom = np.flatnonzero(np.isin(m.geom_bodyid, body))
        site = np.flatnonzero(np.isin(m.site_bodyid, body))
        d = d.replace(**{k: getattr(d, k).at[body].set(getattr(placed, k)[body]) for k in _KINEMATICS[:5]})
        d = d.replace(**{k: getattr(d, k).at[site].set(getattr(placed, k)[site]) for k in ("site_xpos", "site_xmat")})
        geom_xpos, geom_xmat = placed.geom_xpos, placed.geom_xmat
        if len(self._static_geoms):
            # MuJoCo Warp poses geoms welded to the world only when it makes data
            g = self._static_geoms
            xquat = placed.xquat[m.geom_bodyid[g]]
            geom_pos = jnp.asarray(m.geom_pos[g], d.xpos.dtype)
            geom_quat = jnp.asarray(m.geom_quat[g], d.xquat.dtype)
            geom_xpos = geom_xpos.at[g].set(placed.xpos[m.geom_bodyid[g]] + jax.vmap(mjx_math.rotate)(geom_pos, xquat))
            mats = jax.vmap(lambda q, r: mjx_math.quat_to_mat(mjx_math.quat_mul(q, r)))(xquat, geom_quat)
            geom_xmat = geom_xmat.at[g].set(mats.reshape(geom_xmat[g].shape))
        return d.replace(geom_xpos=d.geom_xpos.at[geom].set(geom_xpos[geom]), geom_xmat=d.geom_xmat.at[geom].set(geom_xmat[geom]))

    def forward(self, d, layout):
        return mjx.forward(self._placed(layout), d)

    def kinematics(self, d, layout):
        return mjx.kinematics(self._placed(layout), d)

    def step(self, d, layout, n):
        """`n` simulator steps of world `d`, restarting it as MuJoCo does if it diverged."""
        m = self._placed(layout)
        return self._restart_if_diverged(jax.lax.fori_loop(0, n, lambda _, d: mjx.step(m, d), d))

    def _restart_if_diverged(self, d):
        """`d`, or MuJoCo's `mj_resetData` state if its positions, velocities, or accelerations diverged."""
        diverged = jnp.any(jnp.stack([
            ~jnp.all(jnp.isfinite(x) & (jnp.abs(x) < _MAX_VALUE)) for x in (d.qpos, d.qvel, d.qacc)
        ]))
        m = self.model
        mocap = m.body_mocapid >= 0
        initial = dict(
            qpos=m.qpos0, qvel=np.zeros(m.nv), qacc=np.zeros(m.nv), qacc_warmstart=np.zeros(m.nv), act=np.zeros(m.na),
            ctrl=np.zeros(m.nu), mocap_pos=m.body_pos[mocap][np.argsort(m.body_mocapid[mocap])],
            mocap_quat=m.body_quat[mocap][np.argsort(m.body_mocapid[mocap])],
        )
        return d.replace(**{
            k: jnp.where(diverged, jnp.asarray(v, getattr(d, k).dtype).reshape(getattr(d, k).shape), getattr(d, k))
            for k, v in initial.items()
        })

    def set_state(self, d, layout, qpos=None, qvel=None):
        """Meta-World's `set_state`: `d` at joint positions `qpos` and velocities `qvel`, forwarded."""
        d = d.replace(
            qpos=d.qpos if qpos is None else jnp.asarray(qpos, d.qpos.dtype),
            qvel=d.qvel if qvel is None else jnp.asarray(qvel, d.qvel.dtype),
        )
        return self.forward(d, layout)


class MetaworldPOMDP:
    """A Meta-World task simulated by MJX."""
    def __init__(
        self,
        task: str,
        physics: str | dict = "fast",
        goal_observable: bool = True,
        rand_vecs: jax.Array | None = None,
        camera_names: Sequence[str] = (),
        camera_height: int = 84,
        camera_width: int = 84,
        camera_depths: bool = False,
        observe_state: bool = True,
        observe_proprio: bool = False,
        terminate_on_success: bool = False,
        max_worlds: int = 4096,
        contacts_per_world: int | None = None,
        constraints_per_world: int | None = None,
        graph_mode: Literal["jax", "warp", "none"] = "warp",
    ):
        """
        Simulates with the simulation options of `physics`, a name in `PHYSICS` or a dict of MuJoCo option overrides.
        With `goal_observable`, the last three observations are the goal position, as in Meta-World's MT
        benchmarks; otherwise they are zeros, as in its ML benchmarks. With `rand_vecs`, resets draw uniformly
        from these random reset vectors, as a Meta-World benchmark draws from its goals.

        Without cameras or proprioception, observations are Meta-World's 39-vector. Otherwise they are a dict of
        "state", that vector, with `observe_state`; "proprio", the hand position and gripper opening, and the same at
        the previous step, with `observe_proprio`; "<camera>_image" for each of `camera_names`; and "<camera>_depth" with `camera_depths`.
        With cameras, `observe` is unbatched or mapped over exactly `max_worlds` states. At most `max_worlds` worlds are simulated
        together, with `contacts_per_world` contacts on average and `constraints_per_world` constraint rows each.
        """
        if task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}, not {task!r}")
        if not (observe_state or observe_proprio or camera_names):
            raise ValueError("the POMDP must observe the state, proprioception, or cameras")
        m = load_model(task)
        self.model = m
        self._task = TASKS[task](m)
        self._goal_observable = goal_observable
        self._terminate_on_success = terminate_on_success
        self._rand_vecs = None if rand_vecs is None else jnp.asarray(rand_vecs, jnp.float32)
        default_contacts, default_constraints = _BUFFER_SIZES
        self.max_worlds = max_worlds
        self.sim = Sim(
            m, self._task, PHYSICS[physics] if isinstance(physics, str) else physics,
            naconmax=int(max_worlds * (contacts_per_world or default_contacts)),
            njmax=int(constraints_per_world or default_constraints), graph_mode=graph_mode,
        )
        self._mocap_low = jnp.asarray(self._task.mocap_low, jnp.float32)
        self._mocap_high = jnp.asarray(self._task.mocap_high, jnp.float32)
        self._hand = m.body("hand").id
        self._claws = m.body("rightclaw").id, m.body("leftclaw").id
        self._mocap = m.body_mocapid[m.body("mocap").id]
        goal_low, goal_high = (np.asarray(self._task.goal_low), np.asarray(self._task.goal_high)) if goal_observable \
            else (np.zeros(3), np.zeros(3))
        self._obs_low = jnp.asarray(np.concatenate([_HAND_LOW, [-1.], np.full(_OBJECT_OBS, -np.inf)] * 2 + [goal_low]), jnp.float32)
        self._obs_high = jnp.asarray(np.concatenate([_HAND_HIGH, [1.], np.full(_OBJECT_OBS, np.inf)] * 2 + [goal_high]), jnp.float32)
        # host arrays, so that states never share buffers with the POMDP that donating them would free
        self._fixed_hand_reset = None if self._task.hand_reset_depends_on_layout \
            else jax.tree.map(np.asarray, self.sim.put_data(self._recorded_hand_reset(task)))

        self._observe_state, self._observe_proprio = observe_state, observe_proprio
        self._sites = _visible_sites(m)
        self._markers = np.array([m.geom(f"{_SITE_MARKERS}_{m.site(i).name}").id for i in self._sites], dtype=int)
        self._placed = {m.site(name).id: key for name, key in self._task.placed_sites.items()}
        self._renderer = None
        if camera_names:
            self._renderer = CameraRenderer(
                self.sim.simulated, camera_names=camera_names, height=camera_height, width=camera_width, depth=camera_depths,
                nworld=max_worlds, graph_mode=getattr(wp.JaxCallableGraphMode, graph_mode.upper()),
            )

        self.action_space = pomdps.spaces.Box(-1., 1., (4,))
        state_space = pomdps.spaces.Box(np.asarray(self._obs_low), np.asarray(self._obs_high), (39,))
        if self._renderer is None and not observe_proprio:
            self.observation_space = state_space
        else:
            spaces = {"state": state_space} if observe_state else {}
            if observe_proprio:
                spaces["proprio"] = pomdps.spaces.Box(
                    np.concatenate([_HAND_LOW, [0.]] * 2), np.concatenate([_HAND_HIGH, [1.]] * 2), (8,),
                )
            if self._renderer is not None:
                for name in camera_names:
                    spaces[f"{name}_image"] = pomdps.spaces.Image(camera_height, camera_width, 3)
                    if camera_depths:
                        spaces[f"{name}_depth"] = pomdps.spaces.Box(0., np.inf, (camera_height, camera_width, 1))
            self.observation_space = pomdps.spaces.Dict(spaces)

    @property
    def task(self) -> Task:
        return self._task

    def _recorded_hand_reset(self, task) -> mujoco.MjData:
        """MuJoCo data of upstream Meta-World's hand reset of `task` with the model's layout."""
        m = self.model
        d = mujoco.MjData(m)
        with np.load(_ASSETS / "hand_resets.npz") as resets:
            for k in _HAND_RESET_FIELDS:
                getattr(d, k)[:] = resets[f"{task}/{k}"]
            # the data's kinematics are of the joint positions before the hand reset's last simulator step
            qpos = d.qpos.copy()
            d.qpos[:] = resets[f"{task}/kinematics_qpos"]
            mujoco.mj_forward(m, d)
            d.qpos[:] = qpos
        return d

    def _hand_reset(self, layout):
        """World data at the model's initial state after the hand has moved to its initial position."""
        d = self.sim.make_data(layout)
        d = d.replace(
            mocap_pos=d.mocap_pos.at[self._mocap].set(jnp.asarray(self._task.hand_init_pos, d.mocap_pos.dtype)),
            mocap_quat=d.mocap_quat.at[self._mocap].set(jnp.asarray(_MOCAP_QUAT, d.mocap_quat.dtype)),
            ctrl=jnp.asarray(_HAND_RESET_CTRL, d.ctrl.dtype),
        )
        return self.sim.step(d, layout, _HAND_RESET_STEPS * self.sim.substeps)

    def sample_rand_vec(self, key) -> jax.Array:
        """A random vector of the task's initial object and goal positions, as Meta-World samples them."""
        return self._task.sample(key)

    def init_state(self, rand_vec) -> State:
        """The state at the start of an episode with Meta-World's random vector `rand_vec`."""
        rand_vec = jnp.asarray(rand_vec, jnp.float32)
        layout = self._task.layout(rand_vec, jnp.asarray(self.sim.default_layout))
        if self._fixed_hand_reset is None:
            d = self._hand_reset(layout)
        else:
            d = self.sim.place(jax.tree.map(jnp.asarray, self._fixed_hand_reset), layout)
        d, episode = self._task.reset(self.sim, d, layout, rand_vec)
        # constants a task computes alike share no buffer, so that states can be donated
        episode = jax.tree.map(jnp.copy, episode)
        state = State(d, jnp.zeros(18, jnp.float32), layout, episode)
        return state._replace(prev=self._current(state))

    def reset(self, key):
        if self._rand_vecs is None:
            return self.init_state(self.sample_rand_vec(key))
        return self.init_state(self._rand_vecs[jax.random.randint(key, (), 0, self._rand_vecs.shape[0])])

    def step(self, key, state, action):
        action = jnp.asarray(action, jnp.float32)
        d = state.data
        mocap_pos = d.mocap_pos[self._mocap] + jnp.clip(action[:3], -1., 1.) * _ACTION_SCALE
        d = d.replace(
            mocap_pos=d.mocap_pos.at[self._mocap].set(jnp.clip(mocap_pos, self._mocap_low, self._mocap_high)),
            mocap_quat=d.mocap_quat.at[self._mocap].set(jnp.asarray(_MOCAP_QUAT, d.mocap_quat.dtype)),
            ctrl=jnp.stack([action[3], -action[3]]).astype(d.ctrl.dtype),
        )
        d = self.sim.step(d, state.layout, self.sim.substeps)
        d = self.sim.kinematics(d, state.layout)
        return state._replace(data=d, prev=self._current(state))

    def _current(self, state):
        """Meta-World's unclipped hand, gripper, and object observation of `state`."""
        d = state.data
        right, left = self._claws
        opening = jnp.clip(jnp.linalg.norm(d.xpos[right] - d.xpos[left]) / _GRIPPER_SPAN, 0., 1.)
        objects = jnp.concatenate(self._task.objects(d, state.episode))
        objects = jnp.pad(objects, (0, _OBJECT_OBS - objects.shape[0]))
        return jnp.concatenate([d.xpos[self._hand], opening[None], objects]).astype(jnp.float32)

    def _state_obs(self, state):
        """Meta-World's observation vector of `state`."""
        goal = state.episode["target_pos"] if self._goal_observable else jnp.zeros(3)
        obs = jnp.concatenate([self._current(state), state.prev, goal.astype(jnp.float32)])
        return jnp.clip(obs, self._obs_low, self._obs_high)

    def observe(self, key, next_state, action):
        state = self._state_obs(next_state)
        if not isinstance(self.observation_space, pomdps.spaces.Dict):
            return state
        obs = {"state": state} if self._observe_state else {}
        if self._observe_proprio:
            obs["proprio"] = jnp.concatenate([state[:4], state[18:22]])
        if self._renderer is not None:
            obs |= self._renderer(self._with_site_markers(next_state))
        return obs

    def _with_site_markers(self, state):
        """World data of `state` with the site markers at their sites, the sites the task places where it placed them."""
        d = state.data
        pos = d.site_xpos[self._sites]
        for site, key in self._placed.items():
            body = self.model.site_bodyid[site]
            local = jnp.matmul(d.xmat[body].reshape(3, 3), state.episode[key], precision="highest")
            pos = pos.at[np.flatnonzero(self._sites == site)].set(d.xpos[body] + local)
        return d.replace(geom_xpos=d.geom_xpos.at[self._markers].set(pos.astype(d.geom_xpos.dtype)))

    def evaluate(self, state, action) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Meta-World's reward and info for arriving at `state` by `action`."""
        obs = self._state_obs(state)
        return self._task.evaluate(state.data, obs, jnp.asarray(action, jnp.float32), state.episode)

    def reward(self, key, state, action, next_state):
        return self.evaluate(next_state, action)[0]

    def success(self, state):
        """Whether the task has been accomplished in `state`."""
        return self.evaluate(state, jnp.zeros(4))[1]["success"] > 0.

    def done(self, state):
        return self._terminate_on_success & self.success(state)

    def overflowed(self, state):
        """Whether contact or constraint buffers overflowed while simulating `state`, dropping contacts or constraints."""
        return (state.data._impl.overflow & _BUFFER_OVERFLOWS) != 0


# camera and image size of the pixel observation modes
PIXELS = (("corner2",), 84)
_CAMERAS, _SIZE = PIXELS
# keyword arguments of each observation mode
OBSERVATION_MODES = {
    "mkv": {},
    "pix": dict(camera_names=_CAMERAS, camera_height=_SIZE, camera_width=_SIZE, observe_state=False),
    "pix-prp": dict(
        camera_names=_CAMERAS, camera_height=_SIZE, camera_width=_SIZE, observe_state=False, observe_proprio=True,
    ),
}
DEFAULT_OBSERVATION_MODE = "mkv"

for _task in TASKS:
    for _mode, _kwargs in OBSERVATION_MODES.items():
        pomdps.register(f"{_task}/{_mode}", functools.partial(MetaworldPOMDP, _task, **_kwargs))
    pomdps.register(_task, functools.partial(MetaworldPOMDP, _task, **OBSERVATION_MODES[DEFAULT_OBSERVATION_MODE]))
