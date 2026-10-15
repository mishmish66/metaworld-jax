"""Meta-World's drawer-open-v3: pull a drawer open."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

# how far the drawer opens
MAX_DIST = 0.2


class DrawerOpen(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_drawer.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.9, 0.0)
    rand_high = (0.1, 0.9, 0.0)
    goal_low = hand_low
    goal_high = hand_high
    moved_bodies = ("drawer",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("drawer")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        target_pos = rand_vec + jnp.array([0., -0.16 - MAX_DIST, 0.09])
        return d, dict(target_pos=target_pos, init_tcp=self.tcp_center(d))

    def pos_objects(self, d, episode):
        return self.body_com(d, "drawer_link") + jnp.array([0., -0.16, 0.])

    def quat_objects(self, d, episode):
        return d.xquat[self.body("drawer_link")]

    def evaluate(self, d, obs, action, episode):
        gripper = obs[:3]
        handle = obs[4:7]
        target = episode["target_pos"]
        handle_error = jnp.linalg.norm(handle - target)
        reward_for_opening = rewards.tolerance(handle_error, bounds=(0., 0.02), margin=MAX_DIST, sigmoid="long_tail")
        handle_pos_init = target + jnp.array([0., MAX_DIST, 0.])
        # XY error dominates so the gripper can drop down and cage the handle
        scale = jnp.array([3., 3., 1.])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - episode["init_tcp"]) * scale
        reward_for_caging = rewards.tolerance(
            jnp.linalg.norm(gripper_error), bounds=(0., 0.01), margin=jnp.linalg.norm(gripper_error_init),
            sigmoid="long_tail",
        )
        reward = 5. * (reward_for_caging + reward_for_opening)
        return reward, info(
            success=handle_error <= 0.03, near_object=jnp.linalg.norm(handle - gripper) <= 0.03,
            grasp_success=obs[3] > 0, grasp_reward=reward_for_caging, in_place_reward=reward_for_opening,
            obj_to_target=handle_error, unscaled_reward=reward,
        )
