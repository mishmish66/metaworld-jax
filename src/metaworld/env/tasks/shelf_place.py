"""Meta-World's shelf-place-v3: pick up a block and place it on a shelf."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class ShelfPlace(Task):
    xml = "sawyer_shelf_placing.xml"
    physics = "upstream"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.5, 0.019, -0.1, 0.8, 0.299)
    rand_high = (0.1, 0.6, 0.021, 0.1, 0.9, 0.301)
    goal_low = (-0.1, 0.8, 0.299)
    goal_high = (0.1, 0.9, 0.301)
    moved_bodies = ("shelf",)


    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.1)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("shelf")].set(rand_vec[3:] - jnp.array([0., 0., 0.3]))

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], self.body_com(d, "obj")[2:]])
        shelf = layout[self.layout_index("shelf")]
        target_pos = jnp.asarray(self.m.site_pos[self.site("goal")], jnp.float32) + shelf
        episode = dict(target_pos=target_pos, obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.body_com(d, "obj")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "objGeom")

    def evaluate(self, d, obs, action, episode):
        target_radius = 0.05
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        tcp_opened = obs[3]
        target, obj_init_pos = episode["target_pos"], episode["obj_init_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        in_place_margin = jnp.linalg.norm(obj_init_pos - target)
        in_place = rewards.tolerance(obj_to_target, bounds=(0., target_radius), margin=in_place_margin, sigmoid="long_tail")
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, obj_radius=0.02, pad_success_thresh=0.05, object_reach_radius=0.01,
            xz_thresh=0.01,
        )
        reward = rewards.hamacher_product(object_grasped, in_place)
        below_shelf = (0. < obj[2]) & (obj[2] < 0.24) & (target[0] - 0.15 < obj[0]) & (obj[0] < target[0] + 0.15)
        in_front = below_shelf & (target[1] - 3 * target_radius < obj[1]) & (obj[1] < target[1])
        z_scaling = (0.24 - obj[2]) / 0.24
        y_scaling = (obj[1] - (target[1] - 3 * target_radius)) / (3 * target_radius)
        bound_loss = rewards.hamacher_product(y_scaling, z_scaling)
        in_place = jnp.where(in_front, jnp.clip(in_place - bound_loss, 0., 1.), in_place)
        in_place = jnp.where(below_shelf & (obj[1] > target[1]), 0., in_place)
        lifted = (tcp_to_obj < 0.025) & (tcp_opened > 0) & (obj[2] - 0.01 > obj_init_pos[2])
        reward = jnp.where(lifted, reward + 1. + 5. * in_place, reward)
        reward = jnp.where(obj_to_target < target_radius, 10., reward)
        grasp_success = self.touching_main_object(d) & (tcp_opened > 0) & (obj[2] - 0.02 > obj_init_pos[2])
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
