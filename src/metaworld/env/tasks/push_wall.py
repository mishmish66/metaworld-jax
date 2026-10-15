"""Meta-World's push-wall-v3: push a puck around a wall to a goal position."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class PushWall(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_push_wall_v3.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.05, 0.6, 0.015, -0.05, 0.85, 0.01)
    rand_high = (0.05, 0.65, 0.015, 0.05, 0.9, 0.02)
    goal_low = (-0.05, 0.85, 0.01)
    goal_high = (0.05, 0.9, 0.02)

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

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        tcp_opened = obs[3]
        target, obj_init_pos = episode["target_pos"], episode["obj_init_pos"]
        midpoint = jnp.stack([-0.05, 0.77, obj[2]])
        in_place_scaling = jnp.array([3., 1., 1.])
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        obj_to_midpoint = jnp.linalg.norm((obj - midpoint) * in_place_scaling)
        obj_to_midpoint_init = jnp.linalg.norm((obj_init_pos - midpoint) * in_place_scaling)
        obj_to_target = jnp.linalg.norm(obj - target)
        in_place_part1 = rewards.tolerance(
            obj_to_midpoint, bounds=(0., self.TARGET_RADIUS), margin=obj_to_midpoint_init, sigmoid="long_tail",
        )
        in_place_part2 = rewards.tolerance(
            obj_to_target, bounds=(0., self.TARGET_RADIUS), margin=jnp.linalg.norm(obj_init_pos - target),
            sigmoid="long_tail",
        )
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, object_reach_radius=0.01, obj_radius=0.015, pad_success_thresh=0.05,
            xz_thresh=0.005, high_density=True,
        )
        reward = 2. * object_grasped
        near = (tcp_to_obj < 0.02) & (tcp_opened > 0)
        reward = jnp.where(
            near,
            jnp.where(obj[1] > 0.75, 2. * object_grasped + 5. + 3. * in_place_part2, 2. * object_grasped + 1. + 4. * in_place_part1),
            reward,
        )
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10., reward)
        grasp_success = self.touching_main_object(d) & (tcp_opened > 0) & (obj[2] - 0.02 > obj_init_pos[2])
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place_part2, obj_to_target=obj_to_target,
            unscaled_reward=reward,
        )
