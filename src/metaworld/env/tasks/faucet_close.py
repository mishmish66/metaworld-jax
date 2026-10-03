"""Meta-World's faucet-close-v3: turn a faucet's handle closed."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

HANDLE_LENGTH = 0.175
TARGET_RADIUS = 0.07


class FaucetClose(Task):
    placed_sites = {"goal_close": "target_pos"}
    xml = "sawyer_faucet.xml"
    hand_low = (-0.5, 0.40, -0.15)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (-0.1, 0.8, 0.0)
    rand_high = (0.1, 0.85, 0.0)
    goal_low = hand_low
    goal_high = hand_high
    moved_bodies = ("faucetBase",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("faucetBase")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        target_pos = rand_vec + jnp.array([-HANDLE_LENGTH, 0., 0.125])
        return sim.forward(d, layout), dict(target_pos=target_pos, obj_init_pos=rand_vec, init_tcp=self.tcp_center(d))

    def pos_objects(self, d, episode):
        return self.site_pos(d, "handleStartClose") + jnp.array([0., 0., -0.01])

    def quat_objects(self, d, episode):
        return d.xquat[self.body("faucetBase")]

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        tcp = self.tcp_center(d)
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(obj_init_pos - target)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., TARGET_RADIUS), margin=jnp.abs(target_to_obj_init - TARGET_RADIUS),
            sigmoid="long_tail",
        )
        faucet_reach_radius = 0.01
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        tcp_to_obj_init = jnp.linalg.norm(obj_init_pos - episode["init_tcp"])
        reach = rewards.tolerance(
            tcp_to_obj, bounds=(0., faucet_reach_radius), margin=jnp.abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid="gaussian",
        )
        success = target_to_obj <= TARGET_RADIUS
        reward = jnp.where(success, 10., 2. * (2. * reach + 3. * in_place))
        return reward, info(
            success=success, near_object=tcp_to_obj <= 0.01, grasp_success=1., grasp_reward=reach,
            in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
