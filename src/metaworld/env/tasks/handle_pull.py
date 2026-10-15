"""Meta-World's handle-pull-v3: pull a handle up."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class HandlePull(Task):
    xml = "sawyer_handle_press.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1.0, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.8, -0.001)
    rand_high = (0.1, 0.9, 0.001)
    goal_low = (-0.1, 0.55, 0.04)
    goal_high = (0.1, 0.70, 0.18)
    moved_bodies = ("box",)
    hand_reset_depends_on_layout = True

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("box")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        init_tcp = self.tcp_center(d)
        d = sim.set_state(d, layout, d.qpos.at[9].set(-0.1), d.qvel.at[9].set(0.))
        return d, dict(target_pos=self.site_pos(d, "goalPull"), obj_init_pos=rand_vec, init_tcp=init_tcp)

    def pos_objects(self, d, episode):
        return self.site_pos(d, "handleRight")

    def quat_objects(self, d, episode):
        return jnp.zeros(4)

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        target_to_obj = jnp.abs(target[2] - obj[2])
        target_to_obj_init = jnp.abs(target[2] - obj_init_pos[2])
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=target_to_obj_init, sigmoid="long_tail",
        )
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, pad_success_thresh=0.05, obj_radius=0.022, object_reach_radius=0.01,
            xz_thresh=0.01, high_density=True,
        )
        reward = rewards.hamacher_product(object_grasped, in_place)
        tcp_opened = obs[3]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        # Meta-World compares the handle's y with the initial z
        pulled = (tcp_to_obj < 0.035) & (tcp_opened > 0) & (obj[1] - 0.01 > obj_init_pos[2])
        reward = jnp.where(pulled, reward + 1. + 5. * in_place, reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10., reward)
        return reward, info(
            success=target_to_obj <= self.TARGET_RADIUS, near_object=tcp_to_obj <= 0.05,
            grasp_success=(tcp_opened > 0) & (obj[2] - 0.03 > obj_init_pos[2]), grasp_reward=object_grasped,
            in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
