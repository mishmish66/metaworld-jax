"""Meta-World's soccer-v3: push a ball into a goal."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info
from metaworld.env.tasks.sweep import centered_caging_reward


class Soccer(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_soccer.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.6, 0.03, -0.1, 0.8, 0.0)
    rand_high = (0.1, 0.7, 0.03, 0.1, 0.9, 0.0)
    goal_low = (-0.1, 0.8, 0.0)
    goal_high = (0.1, 0.9, 0.0)
    moved_bodies = ("goal_whole",)
    OBJ_RADIUS = 0.013
    TARGET_RADIUS = 0.07


    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.15)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("goal_whole")].set(rand_vec[3:])

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([0.03])])
        episode = dict(target_pos=rand_vec[3:], obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.body_com(d, "soccer_ball")

    def quat_objects(self, d, episode):
        return self.body_mat_quat(d, "soccer_ball")

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        target, obj_init_pos = episode["target_pos"], episode["obj_init_pos"]
        x_scaling = jnp.array([3., 1., 1.])
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        target_to_obj = jnp.linalg.norm((obj - target) * x_scaling)
        target_to_obj_init = jnp.linalg.norm((obj - obj_init_pos) * x_scaling)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=target_to_obj_init, sigmoid="long_tail",
        )
        goal_line = target[1] - 0.1
        off_goal = (obj[1] > goal_line) & (jnp.abs(obj[0] - target[0]) > 0.10)
        in_place = jnp.where(
            off_goal, jnp.clip(in_place - 2. * ((obj[1] - goal_line) / (1. - goal_line)), 0., 1.), in_place,
        )
        object_grasped = centered_caging_reward(
            self, d, episode, action, obj, self.OBJ_RADIUS, self.OBJ_RADIUS + 0.01, 0.005,
        )
        reward = 3. * object_grasped + 6.5 * in_place
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10., reward)
        obj_to_target = jnp.linalg.norm(obj - target)
        grasp_success = self.touching_main_object(d) & (obs[3] > 0) & (obj[2] - 0.02 > obj_init_pos[2])
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
