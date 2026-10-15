"""Meta-World's drawer-close-v3: push an open drawer closed."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

# how far the drawer starts open
MAX_DIST = 0.15
_HANDLE = jnp.array([0., -0.16, 0.05])


class DrawerClose(Task):
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
        init_tcp = self.tcp_center(d)
        d = sim.set_state(d, layout, d.qpos.at[9].set(-MAX_DIST))
        return d, dict(
            target_pos=rand_vec + jnp.array([0., -0.16, 0.09]), obj_init_pos=self.pos_objects(d, None), init_tcp=init_tcp,
        )

    def pos_objects(self, d, episode):
        return self.body_com(d, "drawer_link") + _HANDLE

    def quat_objects(self, d, episode):
        return jnp.zeros(4)

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        tcp = self.tcp_center(d)
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(obj_init_pos - target)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=jnp.abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        handle_reach_radius = 0.005
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        tcp_to_obj_init = jnp.linalg.norm(obj_init_pos - episode["init_tcp"])
        reach = rewards.tolerance(
            tcp_to_obj, bounds=(0., handle_reach_radius), margin=jnp.abs(tcp_to_obj_init - handle_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = jnp.minimum(jnp.maximum(0., action[-1]), 1.)
        reach = rewards.hamacher_product(reach, gripper_closed)
        success = target_to_obj <= self.TARGET_RADIUS + 0.015
        reward = 10. * jnp.where(success, 1., rewards.hamacher_product(reach, in_place))
        return reward, info(
            success=success, near_object=tcp_to_obj <= 0.01, grasp_success=1., grasp_reward=reach,
            in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
