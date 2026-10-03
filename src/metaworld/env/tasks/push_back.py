"""Meta-World's push-back-v3: push a puck back toward the robot to a goal position."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class PushBack(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_push_back_v3.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.8, 0.02, -0.1, 0.6, 0.0199)
    rand_high = (0.1, 0.85, 0.02, 0.1, 0.7, 0.0201)
    goal_low = (-0.1, 0.6, 0.0199)
    goal_high = (0.1, 0.7, 0.0201)
    OBJ_RADIUS = 0.007

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.15)

    def reset(self, sim, d, layout, rand_vec):
        height = self.geom_pos(d, "objGeom")[2][None]
        target_pos = jnp.concatenate([rand_vec[3:5], height])
        obj_init_pos = jnp.concatenate([rand_vec[:2], height])
        episode = dict(target_pos=target_pos, obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.geom_pos(d, "objGeom")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "objGeom")

    def _gripper_caging_reward(self, d, episode, obj_pos, obj_radius):
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.003
        x_z_success_margin = 0.01
        tcp = self.tcp_center(d)
        left_pad, right_pad = self.body_com(d, "leftpad"), self.body_com(d, "rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_pos[1]
        delta_object_y_right_pad = obj_pos[1] - right_pad[1]
        # Meta-World's initial pad positions are views of the current ones
        right_caging_margin = jnp.abs(jnp.abs(obj_pos[1] - right_pad[1]) - pad_success_margin)
        left_caging_margin = jnp.abs(jnp.abs(obj_pos[1] - left_pad[1]) - pad_success_margin)

        def tolerance(x, upper, margin):
            return rewards.tolerance(x, bounds=(obj_radius, upper), margin=margin, sigmoid="long_tail")

        y_caging = rewards.hamacher_product(
            tolerance(delta_object_y_right_pad, pad_success_margin, right_caging_margin),
            tolerance(delta_object_y_left_pad, pad_success_margin, left_caging_margin),
        )
        y_gripping = rewards.hamacher_product(
            tolerance(delta_object_y_right_pad, grip_success_margin, right_caging_margin),
            tolerance(delta_object_y_left_pad, grip_success_margin, left_caging_margin),
        )
        xz = jnp.array([1., 0., 1.])
        tcp_obj_x_z_margin = jnp.linalg.norm(episode["obj_init_pos"] * xz - episode["init_tcp"] * xz) - x_z_success_margin
        x_z_caging = rewards.tolerance(
            jnp.linalg.norm(tcp * xz - obj_pos * xz), bounds=(0., x_z_success_margin), margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )
        caging = rewards.hamacher_product(y_caging, x_z_caging)
        gripping = jnp.where(caging > 0.95, y_gripping, 0.)
        return (caging + gripping) / 2.

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        tcp_opened = obs[3]
        target, obj_init_pos = episode["target_pos"], episode["obj_init_pos"]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(obj_init_pos - target)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=target_to_obj_init, sigmoid="long_tail",
        )
        object_grasped = self._gripper_caging_reward(d, episode, obj, self.OBJ_RADIUS)
        reward = rewards.hamacher_product(object_grasped, in_place)
        pushing = (
            (tcp_to_obj < 0.01) & (0 < tcp_opened) & (tcp_opened < 0.55) & (target_to_obj_init - target_to_obj > 0.01)
        )
        reward = jnp.where(pushing, reward + 1. + 5. * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10., reward)
        grasp_success = self.touching_main_object(d) & (tcp_opened > 0) & (obj[2] - 0.02 > obj_init_pos[2])
        return reward, info(
            success=target_to_obj <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
