"""Meta-World's peg-insert-side-v3: insert a peg sideways into a hole in a box."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.geometry import mat2quat
from metaworld.env.tasks.base import Task, info


class PegInsertSide(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_peg_insertion_side.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (0., 0.5, 0.02, -0.35, 0.4, -0.001)
    rand_high = (0.2, 0.7, 0.02, -0.25, 0.7, 0.001)
    goal_low = (-0.32, 0.4, 0.129)
    goal_high = (-0.22, 0.7, 0.131)
    moved_bodies = ("box",)
    TARGET_RADIUS = 0.07

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.1)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("box")].set(rand_vec[3:])

    def reset(self, sim, d, layout, rand_vec):
        pos_peg, pos_box = rand_vec[:3], rand_vec[3:]
        episode = dict(
            target_pos=pos_box + jnp.array([0.03, 0., 0.13]), obj_init_pos=pos_peg, init_tcp=self.tcp_center(d),
            peg_head_pos_init=self.site_pos(d, "pegHead"),
        )
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(pos_peg), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.site_pos(d, "pegGrasp")

    def quat_objects(self, d, episode):
        return mat2quat(d.site_xmat[self.site("pegGrasp")].reshape(3, 3))

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        obj_head = self.site_pos(d, "pegHead")
        tcp_opened = obs[3]
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        scale = jnp.array([1., 2., 2.])
        obj_to_target = jnp.linalg.norm((obj_head - target) * scale)
        in_place_margin = jnp.linalg.norm((episode["peg_head_pos_init"] - target) * scale)
        in_place = rewards.tolerance(obj_to_target, bounds=(0., self.TARGET_RADIUS), margin=in_place_margin, sigmoid="long_tail")
        collision_boxes = rewards.hamacher_product(
            rewards.rect_prism_tolerance(
                obj_head, zero=self.site_pos(d, "bottom_right_corner_collision_box_2"),
                one=self.site_pos(d, "top_left_corner_collision_box_2"),
            ),
            rewards.rect_prism_tolerance(
                obj_head, zero=self.site_pos(d, "bottom_right_corner_collision_box_1"),
                one=self.site_pos(d, "top_left_corner_collision_box_1"),
            ),
        )
        in_place = rewards.hamacher_product(in_place, collision_boxes)
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, object_reach_radius=0.01, obj_radius=0.0075, pad_success_thresh=0.03,
            xz_thresh=0.005, high_density=True,
        )
        lifted = (tcp_to_obj < 0.08) & (tcp_opened > 0) & (obj[2] - 0.01 > obj_init_pos[2])
        object_grasped = jnp.where(lifted, 1., object_grasped)
        reward = rewards.hamacher_product(object_grasped, in_place)
        reward = jnp.where(lifted, reward + 1. + 5. * in_place, reward)
        reward = jnp.where(obj_to_target <= 0.07, 10., reward)
        grasp_success = (tcp_to_obj < 0.02) & (tcp_opened > 0) & (obj[2] - 0.01 > obj_init_pos[2])
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
