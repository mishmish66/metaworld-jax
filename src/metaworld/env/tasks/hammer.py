"""Meta-World's hammer-v3: hammer a nail into a box."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

HAMMER_HANDLE_LENGTH = 0.14


class Hammer(Task):
    xml = "sawyer_hammer.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (-0.1, 0.4, 0.)
    rand_high = (0.1, 0.5, 0.)
    goal_low = (0.2399, 0.7399, 0.109)
    goal_high = (0.2401, 0.7401, 0.111)

    def reset(self, sim, d, layout, rand_vec):
        episode = dict(target_pos=self.site_pos(d, "goal"), obj_init_pos=rand_vec, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(rand_vec), d.qvel.at[9:15].set(0.))
        return d, episode

    def _main_object_geom(self):
        return self.geom("HammerHandle")

    def pos_objects(self, d, episode):
        return jnp.concatenate([self.body_com(d, "hammer"), self.body_com(d, "nail_link")])

    def quat_objects(self, d, episode):
        return jnp.concatenate([d.xquat[self.body("hammer")], d.xquat[self.body("nail_link")]])

    def evaluate(self, d, obs, action, episode):
        hand, hammer = obs[:3], obs[4:7]
        hammer_head = hammer + jnp.array([0.16, 0.06, 0.])
        hammer_threshed = hammer.at[0].set(
            jnp.where(jnp.abs(hammer[0] - hand[0]) < HAMMER_HANDLE_LENGTH / 2., hand[0], hammer[0])
        )
        quat = jnp.maximum(1. - jnp.linalg.norm(obs[7:11] - jnp.array([1., 0., 0., 0.])) / 0.4, 0.)
        reward_grab = self.gripper_caging_reward(
            d, episode, action, hammer_threshed, object_reach_radius=0.01, obj_radius=0.015,
            pad_success_thresh=0.02, xz_thresh=0.01, high_density=True,
        )
        in_place = 0.1 * (hammer_head[2] > 0.02) + 0.9 * rewards.tolerance(
            jnp.linalg.norm(episode["target_pos"] - hammer_head), bounds=(0., 0.02), margin=0.2, sigmoid="long_tail",
        )
        reward = (2. * reward_grab + 6. * in_place) * quat
        success = d.qpos[self.qpos_adr("NailSlideJoint")] > 0.09
        reward = jnp.where(success & (reward > 5.), 10., reward)
        return reward, info(
            success=success, near_object=quat, grasp_success=reward_grab >= 0.5, grasp_reward=reward_grab,
            in_place_reward=in_place, obj_to_target=0., unscaled_reward=reward,
        )
