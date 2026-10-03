"""Meta-World's pick-out-of-hole-v3: pick a peg out of a hole and lift it to a goal position."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class PickOutOfHole(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_pick_out_of_hole.xml"
    hand_low = (-0.5, 0.40, -0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (0, 0.75, 0.02, -0.1, 0.5, 0.15)
    rand_high = (0, 0.75, 0.02, 0.1, 0.6, 0.3)
    goal_low = (-0.1, 0.5, 0.15)
    goal_high = (0.1, 0.6, 0.3)

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
        return d.xquat[self.body("obj")]

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        gripper = self.tcp_center(d)
        target, obj_init_pos = episode["target_pos"], episode["obj_init_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        tcp_to_obj = jnp.linalg.norm(obj - gripper)
        in_place_margin = jnp.linalg.norm(obj_init_pos - target)
        threshold = 0.03
        radius = jnp.linalg.norm(gripper[:2] - obj_init_pos[:2])
        floor = jnp.where(radius <= threshold, 0., 0.015 * jnp.log(jnp.maximum(radius - threshold, 1e-30)) + 0.15)
        above_floor = jnp.where(
            gripper[2] >= floor, 1.,
            rewards.tolerance(jnp.maximum(floor - gripper[2], 0.), bounds=(0., 0.01), margin=0.02, sigmoid="long_tail"),
        )
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, obj_radius=0.015, pad_success_thresh=0.02, object_reach_radius=0.01,
            xz_thresh=0.03, desired_gripper_effort=0.1, high_density=True,
        )
        in_place = rewards.tolerance(obj_to_target, bounds=(0., 0.02), margin=in_place_margin, sigmoid="long_tail")
        reward = rewards.hamacher_product(object_grasped, in_place)
        grasp_success = (tcp_to_obj < 0.04) & (obj[2] - 0.02 > obj_init_pos[2]) & ~(obs[3] < 0.33)
        reward = jnp.where(grasp_success, reward + 1. + 5. * rewards.hamacher_product(in_place, above_floor), reward)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10., reward)
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
