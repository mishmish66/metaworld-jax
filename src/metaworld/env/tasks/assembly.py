"""Meta-World's assembly-v3: pick up a nut and place it onto a peg."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

WRENCH_HANDLE_LENGTH = 0.02


def reward_quat(obs):
    """How close the nut's orientation is to the one that fits the peg."""
    error = jnp.linalg.norm(obs[7:11] - jnp.array([0.707, 0., 0., 0.707]))
    return jnp.maximum(1. - error / 0.4, 0.)


def wrench_threshed(obs):
    """The nut's handle position, with its x snapped to the hand's when the hand is within half a handle of it."""
    hand, wrench = obs[:3], obs[4:7]
    return wrench.at[0].set(jnp.where(jnp.abs(wrench[0] - hand[0]) < WRENCH_HANDLE_LENGTH / 2., hand[0], wrench[0]))


class Assembly(Task):
    placed_sites = {"pegTop": "target_pos"}
    xml = "sawyer_assembly_peg.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (0., 0.6, 0.02, -0.1, 0.75, 0.1)
    rand_high = (0., 0.6, 0.02, 0.1, 0.85, 0.1)
    goal_low = (-0.1, 0.75, 0.1)
    goal_high = (0.1, 0.85, 0.1)
    moved_bodies = ("peg",)

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.1)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("peg")].set(rand_vec[3:] - jnp.array([0., 0., 0.05]))

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos, target_pos = rand_vec[:3], rand_vec[3:]
        episode = dict(target_pos=target_pos, obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def _main_object_geom(self):
        return self.geom("WrenchHandle")

    def pos_objects(self, d, episode):
        return self.site_pos(d, "RoundNut-8")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("RoundNut")]

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos - wrench_center
        radius = jnp.linalg.norm(pos_error[:2])
        success = (radius < 0.02) & (pos_error[2] > 0.)
        threshold = jnp.where(success, 0.02, 0.01)
        target_height = jnp.where(
            radius > threshold, 0.02 * jnp.log(jnp.maximum(radius - threshold, 1e-30)) + 0.2, 0.,
        )
        pos_error = pos_error.at[2].set(target_height - wrench_center[2])
        lifted = (wrench_center[2] > 0.02) | (radius < threshold)
        in_place = 0.1 * lifted + 0.9 * rewards.tolerance(
            jnp.linalg.norm(pos_error * jnp.array([1., 1., 3.])), bounds=(0., 0.02), margin=0.4, sigmoid="long_tail",
        )
        return in_place, success

    def evaluate(self, d, obs, action, episode):
        quat = reward_quat(obs)
        reward_grab = self.gripper_caging_reward(
            d, episode, action, wrench_threshed(obs), object_reach_radius=0.01, obj_radius=0.015,
            pad_success_thresh=0.02, xz_thresh=0.01, medium_density=True,
        )
        in_place, success = self._reward_pos(self.site_pos(d, "RoundNut"), episode["target_pos"])
        reward = jnp.where(success, 10., (2. * reward_grab + 6. * in_place) * quat)
        return reward, info(
            success=success, near_object=quat, grasp_success=reward_grab >= 0.5, grasp_reward=reward_grab,
            in_place_reward=in_place, obj_to_target=0., unscaled_reward=reward,
        )
