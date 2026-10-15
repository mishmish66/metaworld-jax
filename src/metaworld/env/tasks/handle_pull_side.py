"""Meta-World's handle-pull-side-v3: pull up a handle that faces sideways."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class HandlePullSide(Task):
    xml = "sawyer_handle_press_sideways.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1.0, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.35, 0.65, 0.0)
    rand_high = (-0.25, 0.75, 0.0)
    goal_low = hand_low
    goal_high = hand_high
    moved_bodies = ("box",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("box")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        init_tcp = self.tcp_center(d)
        d = sim.set_state(d, layout, d.qpos.at[9].set(-0.1), d.qvel.at[9].set(0.))
        return d, dict(target_pos=self.site_pos(d, "goalPull"), obj_init_pos=self.pos_objects(d, None), init_tcp=init_tcp)

    def pos_objects(self, d, episode):
        return self.site_pos(d, "handleCenter")

    def quat_objects(self, d, episode):
        return jnp.zeros(4)

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
            d, episode, action, obj, pad_success_thresh=0.06, obj_radius=0.032, object_reach_radius=0.01,
            xz_thresh=0.01, high_density=True,
        )
        reward = rewards.hamacher_product(object_grasped, in_place)
        tcp_opened = obs[3]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        pulled = (tcp_to_obj < 0.035) & (tcp_opened > 0) & (obj[2] - 0.01 > obj_init_pos[2])
        reward = jnp.where(pulled, reward + 1. + 5. * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10., reward)
        return reward, info(
            success=target_to_obj <= 0.08, near_object=tcp_to_obj <= 0.05,
            grasp_success=(tcp_opened > 0) & (obj[2] - 0.03 > obj_init_pos[2]), grasp_reward=object_grasped,
            in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
