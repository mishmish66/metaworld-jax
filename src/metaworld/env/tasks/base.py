"""
The base of Meta-World's tasks: model lookups, Meta-World's reset sampling, observation layout, and reward
helpers shared by the tasks, as JAX functions of simulated world data.
"""
import functools

import numpy as np

import jax
import jax.numpy as jnp
import mujoco

from metaworld.env import rewards
from metaworld.env.geometry import ContactTest, mat2quat

# keys of the info dict Meta-World's tasks report
INFO_KEYS = (
    "success", "near_object", "grasp_success", "grasp_reward", "in_place_reward", "obj_to_target", "unscaled_reward",
)


def info(**values) -> dict:
    """Meta-World's info dict from `values` of its `INFO_KEYS`."""
    if set(values) != set(INFO_KEYS):
        raise ValueError(f"info needs exactly {INFO_KEYS}, not {sorted(values)}")
    return {k: jnp.asarray(values[k], jnp.float32) for k in INFO_KEYS}


class Task:
    """
    A Meta-World task. Subclasses set the class attributes, mirror Meta-World's `reset_model` in `reset`, its
    `_get_pos_objects` and `_get_quat_objects` in `pos_objects` and `quat_objects`, and its `evaluate_state` in
    `evaluate`.
    """
    # model file under assets/sawyer_xyz
    xml: str
    hand_low = (-0.2, 0.55, 0.05)
    hand_high = (0.2, 0.75, 0.3)
    # bounds of the mocap body, the hand bounds unless set
    mocap_low: tuple | None = None
    mocap_high: tuple | None = None
    hand_init_pos: tuple
    # bounds of Meta-World's random reset vector
    rand_low: tuple
    rand_high: tuple
    # bounds of the goal position
    goal_low: tuple
    goal_high: tuple
    # bodies whose positions the task sets at reset; they hold their place in the model for other tasks
    moved_bodies: tuple[str, ...] = ()
    # whether the arm or free objects end the hand reset differently with the moved bodies placed
    hand_reset_depends_on_layout = False
    # sites the task places at reset, by the episode constant that is their position in their body
    placed_sites: dict[str, str] = {}
    TARGET_RADIUS = 0.05

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        if self.mocap_low is None:
            self.mocap_low = self.hand_low
        if self.mocap_high is None:
            self.mocap_high = self.hand_high
        self._left_pad, self._right_pad = self.body("leftpad"), self.body("rightpad")
        self._pad_geoms = [self.geom("leftpad_geom"), self.geom("rightpad_geom")]
        # data of the model's initial state, which Meta-World reads some constants from at construction
        self.init_data = mujoco.MjData(m)
        mujoco.mj_forward(m, self.init_data)

    # model lookups

    def body(self, name):
        return self.m.body(name).id

    def site(self, name):
        return self.m.site(name).id

    def geom(self, name):
        return self.m.geom(name).id

    def qpos_adr(self, joint):
        return int(self.m.joint(joint).qposadr[0])

    def dof_adr(self, joint):
        return int(self.m.joint(joint).dofadr[0])

    def layout_index(self, body):
        """Row of `body` in a layout of the moved bodies."""
        return self.moved_bodies.index(body)

    # world data

    def body_com(self, d, name):
        """Meta-World's `get_body_com`, which is the body frame's position."""
        return d.xpos[self.body(name)]

    def site_pos(self, d, name):
        return d.site_xpos[self.site(name)]

    def geom_pos(self, d, name):
        return d.geom_xpos[self.geom(name)]

    def geom_quat(self, d, name):
        """Orientation of the geom as scipy's quaternion (x, y, z, w)."""
        return mat2quat(d.geom_xmat[self.geom(name)].reshape(3, 3))

    def body_mat_quat(self, d, name):
        """Orientation of the body as scipy's quaternion (x, y, z, w)."""
        return mat2quat(d.xmat[self.body(name)].reshape(3, 3))

    def tcp_center(self, d):
        """Midpoint of the two fingers' end effector sites."""
        return (self.site_pos(d, "rightEndEffector") + self.site_pos(d, "leftEndEffector")) / 2.

    @functools.cached_property
    def _main_object_contact(self):
        return [ContactTest(self.m, [pad], [self._main_object_geom()]) for pad in self._pad_geoms]

    def _main_object_geom(self):
        return self.geom("objGeom")

    def touching_main_object(self, d):
        """Whether both finger pads touch the task's main object."""
        return self._main_object_contact[0](d) & self._main_object_contact[1](d)

    # reset

    def uniform(self, key):
        """A random reset vector uniform in its bounds."""
        low, high = (jnp.asarray(b, jnp.float32) for b in (self.rand_low, self.rand_high))
        return jax.random.uniform(key, low.shape, minval=low, maxval=high)

    def sample(self, key):
        """Meta-World's random reset vector."""
        return self.uniform(key)

    def sample_until(self, key, accept):
        """A uniform random reset vector, redrawn until `accept` holds for it."""
        def cond(carry):
            return ~accept(carry[1])

        def body(carry):
            key, _ = carry
            key, sub = jax.random.split(key)
            return key, self.uniform(sub)

        key, sub = jax.random.split(key)
        return jax.lax.while_loop(cond, body, (key, self.uniform(sub)))[1]

    def layout(self, rand_vec, layout):
        """Positions of the moved bodies for `rand_vec`, from the model's `layout`."""
        return layout

    def reset(self, sim, d, layout, rand_vec):
        """World data and episode constants after Meta-World's `reset_model` from data `d` after the hand reset."""
        raise NotImplementedError

    # observations

    def pos_objects(self, d, episode):
        raise NotImplementedError

    def quat_objects(self, d, episode):
        raise NotImplementedError

    def objects(self, d, episode):
        """Position and quaternion of each object, object by object, as Meta-World observes them."""
        pos = jnp.reshape(self.pos_objects(d, episode), (-1, 3))
        quat = jnp.reshape(self.quat_objects(d, episode), (-1, 4))
        return [x for p, q in zip(pos, quat) for x in (p, q)]

    # rewards

    def evaluate(self, d, obs, action, episode):
        """Meta-World's reward and info dict at data `d` with observation `obs` after `action`."""
        raise NotImplementedError

    def gripper_caging_reward(
        self, d, episode, action, obj_pos, obj_radius, pad_success_thresh, object_reach_radius, xz_thresh,
        desired_gripper_effort=1., high_density=False, medium_density=False,
    ):
        """Meta-World's `SawyerXYZEnv._gripper_caging_reward`, with `obj_init_pos` and `init_tcp` from `episode`."""
        if high_density and medium_density:
            raise ValueError("can only be either high_density or medium_density")
        obj_init_pos, init_tcp = episode["obj_init_pos"], episode["init_tcp"]
        pad_y_lr = jnp.stack([self.body_com(d, "leftpad")[1], self.body_com(d, "rightpad")[1]])
        pad_to_obj_lr = jnp.abs(pad_y_lr - obj_pos[1])
        pad_to_objinit_lr = jnp.abs(pad_y_lr - obj_init_pos[1])
        caging_lr_margin = jnp.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            rewards.tolerance(
                pad_to_obj_lr[i], bounds=(obj_radius, pad_success_thresh), margin=caging_lr_margin[i],
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = rewards.hamacher_product(*caging_lr)

        tcp = self.tcp_center(d)
        xz = np.array([0, 2])
        caging_xz_margin = jnp.linalg.norm(obj_init_pos[xz] - init_tcp[xz]) - xz_thresh
        caging_xz = rewards.tolerance(
            jnp.linalg.norm(tcp[xz] - obj_pos[xz]), bounds=(0., xz_thresh), margin=caging_xz_margin,
            sigmoid="long_tail",
        )

        gripper_closed = jnp.minimum(jnp.maximum(0., action[-1]), desired_gripper_effort) / desired_gripper_effort
        caging = rewards.hamacher_product(caging_y, caging_xz)
        gripping = jnp.where(caging > 0.97, gripper_closed, 0.)
        caging_and_gripping = rewards.hamacher_product(caging, gripping)
        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2.
        if medium_density:
            tcp_to_obj = jnp.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = jnp.linalg.norm(obj_init_pos - init_tcp)
            reach_margin = jnp.abs(tcp_to_obj_init - object_reach_radius)
            reach = rewards.tolerance(
                tcp_to_obj, bounds=(0., object_reach_radius), margin=reach_margin, sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2.
        return caging_and_gripping
