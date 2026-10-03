"""Meta-World's bin-picking-v3: move a cube from one bin to the other."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class BinPicking(Task):
    xml = "sawyer_bin_picking.xml"
    hand_low = (-0.5, 0.40, 0.07)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.21, 0.65, 0.02, 0.1199, 0.699, -0.001)
    rand_high = (-0.03, 0.75, 0.02, 0.1201, 0.701, 0.001)
    goal_low = (0.1199, 0.699, -0.001)
    goal_high = (0.1201, 0.701, 0.001)

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], self.body_com(d, "obj")[2:]])
        init_tcp = self.tcp_center(d)
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        target_pos = self.body_com(d, "bin_goal")
        # Meta-World takes the margin from the first step's distance, which the cube at rest keeps
        target_to_obj_init = jnp.linalg.norm(self.body_com(d, "obj") - target_pos)
        return d, dict(
            target_pos=target_pos, obj_init_pos=obj_init_pos, init_tcp=init_tcp, target_to_obj_init=target_to_obj_init,
        )

    def pos_objects(self, d, episode):
        return self.body_com(d, "obj")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("obj")]

    def evaluate(self, d, obs, action, episode):
        hand, obj = obs[:3], obs[4:7]
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=episode["target_to_obj_init"], sigmoid="long_tail",
        )
        threshold = 0.03
        radii = jnp.stack([jnp.linalg.norm(hand[:2] - obj_init_pos[:2]), jnp.linalg.norm(hand[:2] - target[:2])])
        floor = jnp.min(jnp.where(radii > threshold, 0.02 * jnp.log(jnp.maximum(radii - threshold, 1e-30)) + 0.2, 0.))
        above_floor = jnp.where(
            hand[2] >= floor, 1.,
            rewards.tolerance(jnp.maximum(floor - hand[2], 0.), bounds=(0., 0.01), margin=0.05, sigmoid="long_tail"),
        )
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, obj_radius=0.015, pad_success_thresh=0.05, object_reach_radius=0.01,
            xz_thresh=0.01, desired_gripper_effort=0.7, high_density=True,
        )
        reward = rewards.hamacher_product(object_grasped, in_place)
        near_object = jnp.linalg.norm(obj - hand) < 0.04
        grasp_success = near_object & (obj[2] - 0.02 > obj_init_pos[2]) & ~(obs[3] < 0.43)
        reward = jnp.where(grasp_success, reward + 1. + 5. * rewards.hamacher_product(above_floor, in_place), reward)
        reward = jnp.where(target_to_obj < self.TARGET_RADIUS, 10., reward)
        return reward, info(
            success=target_to_obj <= 0.05, near_object=near_object, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
