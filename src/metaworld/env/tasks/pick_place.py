"""Meta-World's pick-place-v3: pick up a puck and place it at a goal position."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class PickPlace(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_pick_place_v3.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.6, 0.02, -0.1, 0.8, 0.05)
    rand_high = (0.1, 0.7, 0.02, 0.1, 0.9, 0.3)
    goal_low = (-0.1, 0.8, 0.05)
    goal_high = (0.1, 0.9, 0.3)

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.15)

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos, target_pos = rand_vec[:3], rand_vec[3:]
        episode = dict(target_pos=target_pos, obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.body_com(d, "obj")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "objGeom")

    def _gripper_caging_reward(self, d, episode, action, obj_pos):
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center(d)
        left_pad, right_pad = self.body_com(d, "leftpad"), self.body_com(d, "rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        # Meta-World's initial pad positions are views of the current ones
        right_caging_margin = jnp.abs(jnp.abs(obj_pos[1] - right_pad[1]) - pad_success_margin)
        left_caging_margin = jnp.abs(jnp.abs(obj_pos[1] - left_pad[1]) - pad_success_margin)
        right_caging = rewards.tolerance(
            delta_object_y_right_pad, bounds=(obj_radius, pad_success_margin), margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = rewards.tolerance(
            delta_object_y_left_pad, bounds=(obj_radius, pad_success_margin), margin=left_caging_margin,
            sigmoid="long_tail",
        )
        y_caging = rewards.hamacher_product(left_caging, right_caging)
        xz = jnp.array([1., 0., 1.])
        tcp_obj_norm_x_z = jnp.linalg.norm(tcp * xz - obj_pos * xz)
        tcp_obj_x_z_margin = jnp.linalg.norm(episode["obj_init_pos"] * xz - episode["init_tcp"] * xz) - x_z_success_margin
        x_z_caging = rewards.tolerance(
            tcp_obj_norm_x_z, bounds=(0., x_z_success_margin), margin=tcp_obj_x_z_margin, sigmoid="long_tail",
        )
        gripper_closed = jnp.minimum(jnp.maximum(0., action[-1]), 1.)
        caging = rewards.hamacher_product(y_caging, x_z_caging)
        gripping = jnp.where(caging > 0.97, gripper_closed, 0.)
        caging_and_gripping = rewards.hamacher_product(caging, gripping)
        return (caging_and_gripping + caging) / 2.

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        in_place_margin = jnp.linalg.norm(obj_init_pos - target)
        in_place = rewards.tolerance(obj_to_target, bounds=(0., self.TARGET_RADIUS), margin=in_place_margin, sigmoid="long_tail")
        object_grasped = self._gripper_caging_reward(d, episode, action, obj)
        reward = rewards.hamacher_product(object_grasped, in_place)
        lifted = (tcp_to_obj < 0.02) & (tcp_opened > 0) & (obj[2] - 0.01 > obj_init_pos[2])
        reward = jnp.where(lifted, reward + 1. + 5. * in_place, reward)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10., reward)
        grasp_success = self.touching_main_object(d) & (tcp_opened > 0) & (obj[2] - 0.02 > obj_init_pos[2])
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
