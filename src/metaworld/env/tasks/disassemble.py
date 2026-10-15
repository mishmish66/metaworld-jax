"""Meta-World's disassemble-v3: lift a nut off a peg."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.assembly import reward_quat, wrench_threshed
from metaworld.env.tasks.base import Task, info


class Disassemble(Task):
    placed_sites = {"pegTop": "peg_top_pos"}
    xml = "sawyer_assembly_peg.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (0., 0.6, 0.025, -0.1, 0.6, 0.1699)
    rand_high = (0.1, 0.75, 0.02501, 0.1, 0.75, 0.1701)
    goal_low = (-0.1, 0.6, 0.1749)
    goal_high = (0.1, 0.75, 0.1751)
    moved_bodies = ("peg",)
    hand_reset_depends_on_layout = True

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.1)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("peg")].set(rand_vec[:3] + jnp.array([0., 0., 0.03]))

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = rand_vec[:3]
        episode = dict(
            target_pos=obj_init_pos + jnp.array([0., 0., 0.15]), obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d),
            peg_top_pos=obj_init_pos + jnp.array([0., 0., 0.08]),
        )
        d = sim.forward(d, layout)
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def _main_object_geom(self):
        return self.geom("WrenchHandle")

    def pos_objects(self, d, episode):
        return self.site_pos(d, "RoundNut-8")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("RoundNut")]

    def evaluate(self, d, obs, action, episode):
        target = episode["target_pos"]
        quat = reward_quat(obs)
        reward_grab = self.gripper_caging_reward(
            d, episode, action, wrench_threshed(obs), object_reach_radius=0.01, obj_radius=0.015,
            pad_success_thresh=0.02, xz_thresh=0.01, high_density=True,
        )
        wrench_center = self.site_pos(d, "RoundNut")
        in_place = 0.1 * (wrench_center[2] > 0.02) + 0.9 * rewards.tolerance(
            jnp.linalg.norm(target + jnp.array([0., 0., 0.1]) - wrench_center), bounds=(0., 0.02), margin=0.2,
            sigmoid="long_tail",
        )
        success = obs[6] > target[2]
        reward = jnp.where(success, 10., (2. * reward_grab + 6. * in_place) * quat)
        return reward, info(
            success=success, near_object=quat, grasp_success=reward_grab >= 0.5, grasp_reward=reward_grab,
            in_place_reward=in_place, obj_to_target=0., unscaled_reward=reward,
        )
