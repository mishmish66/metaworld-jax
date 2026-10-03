"""Meta-World's hand-insert-v3: push a block into a hole in the table."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class HandInsert(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_table_with_hole.xml"
    hand_low = (-0.5, 0.40, -0.15)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.6, 0.05, -0.04, 0.8, -0.0201)
    rand_high = (0.1, 0.7, 0.05, 0.04, 0.88, -0.0199)
    goal_low = (-0.04, 0.8, -0.0201)
    goal_high = (0.04, 0.88, -0.0199)
    # height of the block at reset
    obj_init_z = 0.05

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.15)

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([self.obj_init_z])])
        episode = dict(target_pos=rand_vec[3:], obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.body_com(d, "obj")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("obj")]

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(obj_init_pos - target)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=target_to_obj_init, sigmoid="long_tail",
        )
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, object_reach_radius=0.01, obj_radius=0.015, pad_success_thresh=0.05,
            xz_thresh=0.005, high_density=True,
        )
        reward = rewards.hamacher_product(object_grasped, in_place)
        tcp_opened = obs[3]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        reward = jnp.where((tcp_to_obj < 0.02) & (tcp_opened > 0), reward + 1. + 7. * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10., reward)
        grasp_success = self.touching_main_object(d) & (tcp_opened > 0) & (obj[2] - 0.02 > obj_init_pos[2])
        return reward, info(
            success=target_to_obj <= 0.05, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
