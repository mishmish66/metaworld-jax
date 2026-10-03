"""Meta-World's lever-pull-v3: lift a lever until it points straight up."""
import numpy as np

import jax.numpy as jnp

from mujoco.mjx._src import math as mjx_math

from metaworld.env import rewards
from metaworld.env.geometry import mat2quat
from metaworld.env.tasks.base import Task, info

# objGeom's frame in its body as MuJoCo 3.3.0 compiles its mesh, which Meta-World observes
_OBJ_GEOM_QUAT = (0.662500038723091, 0.662500038723091, -0.24717139537556432, 0.24717139537556432)

LEVER_RADIUS = 0.2
# offset from the fingers to the gripper's shoulder
_SHOULDER = jnp.array([0., 0.055, 0.07])


class LeverPull(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_lever_pull.xml"
    hand_low = (-0.5, 0.40, -0.15)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (-0.1, 0.7, 0.0)
    rand_high = (0.1, 0.8, 0.0)
    goal_low = hand_low
    goal_high = hand_high
    moved_bodies = ("lever",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("lever")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        return d, dict(
            target_pos=rand_vec + jnp.array([0.12, 0., 0.25 + LEVER_RADIUS]),
            lever_pos_init=rand_vec + jnp.array([0.12, -LEVER_RADIUS, 0.25]),
            init_tcp=self.tcp_center(d),
        )

    def pos_objects(self, d, episode):
        return self.site_pos(d, "leverStart")

    def quat_objects(self, d, episode):
        body = self.m.geom_bodyid[self.geom("objGeom")]
        return mat2quat(jnp.matmul(d.xmat[body].reshape(3, 3), mjx_math.quat_to_mat(jnp.asarray(_OBJ_GEOM_QUAT)), precision="highest"))

    def evaluate(self, d, obs, action, episode):
        gripper = obs[:3]
        lever = obs[4:7]
        lever_pos_init = episode["lever_pos_init"]
        # y error is de-emphasized so the shoulder gets under the lever before bumping it
        scale = jnp.array([4., 1., 4.])
        shoulder_to_lever = jnp.linalg.norm((gripper + _SHOULDER - lever) * scale)
        shoulder_to_lever_init = jnp.linalg.norm((episode["init_tcp"] + _SHOULDER - lever_pos_init) * scale)
        ready_to_lift = rewards.tolerance(
            shoulder_to_lever, bounds=(0., 0.02), margin=shoulder_to_lever_init, sigmoid="long_tail",
        )
        lever_angle = -d.qpos[self.qpos_adr("LeverAxis")]
        lever_error = jnp.abs(lever_angle - np.pi / 2.)
        lever_engagement = rewards.tolerance(
            lever_error, bounds=(0., np.pi / 48.), margin=np.pi / 2. - np.pi / 12., sigmoid="long_tail",
        )
        target = episode["target_pos"]
        obj_to_target = jnp.linalg.norm(lever - target)
        in_place_margin = jnp.linalg.norm(lever_pos_init - target)
        in_place = rewards.tolerance(obj_to_target, bounds=(0., 0.04), margin=in_place_margin, sigmoid="long_tail")
        reward = 10. * rewards.hamacher_product(ready_to_lift, in_place)
        return reward, info(
            success=lever_error <= np.pi / 24, near_object=shoulder_to_lever < 0.03, grasp_success=ready_to_lift > 0.9,
            grasp_reward=ready_to_lift, in_place_reward=lever_engagement, obj_to_target=shoulder_to_lever,
            unscaled_reward=reward,
        )
