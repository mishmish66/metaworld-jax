"""Meta-World's basketball-v3: dunk a ball into a hoop."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class Basketball(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_basketball.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.6, 0.0299, -0.1, 0.85, 0.)
    rand_high = (0.1, 0.7, 0.0301, 0.1, 0.9 + 1e-7, 0.)
    goal_low = (-0.1, 0.767, 0.2499)
    goal_high = (0.1, 0.817 + 1e-7, 0.2501)
    moved_bodies = ("basket_goal",)
    TARGET_RADIUS = 0.08

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.15)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("basket_goal")].set(rand_vec[3:])

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([0.03])])
        episode = dict(target_pos=self.site_pos(d, "goal"), obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.body_com(d, "bsktball")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("bsktball")]

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        obj_init_pos = episode["obj_init_pos"]
        target = episode["target_pos"].at[2].set(0.3)
        scale = jnp.array([1., 1., 2.])
        target_to_obj = jnp.linalg.norm((obj - target) * scale)
        target_to_obj_init = jnp.linalg.norm((obj_init_pos - target) * scale)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=target_to_obj_init, sigmoid="long_tail",
        )
        tcp_opened = obs[3]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, object_reach_radius=0.01, obj_radius=0.025, pad_success_thresh=0.06,
            xz_thresh=0.005, high_density=True,
        )
        lifted = (tcp_to_obj < 0.035) & (tcp_opened > 0) & (obj[2] - 0.01 > obj_init_pos[2])
        object_grasped = jnp.where(lifted, 1., object_grasped)
        reward = rewards.hamacher_product(object_grasped, in_place)
        reward = jnp.where(lifted, reward + 1. + 5. * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10., reward)
        return reward, info(
            success=target_to_obj <= self.TARGET_RADIUS, near_object=tcp_to_obj <= 0.05,
            grasp_success=(tcp_opened > 0) & (obj[2] - 0.03 > obj_init_pos[2]), grasp_reward=object_grasped,
            in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
