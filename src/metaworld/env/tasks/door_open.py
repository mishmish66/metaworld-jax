"""Meta-World's door-open-v3: open a door with a revolving joint."""
import numpy as np

import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class DoorOpen(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_door_pull.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (0.0, 0.85, 0.15)
    rand_high = (0.1, 0.95, 0.15)
    goal_low = (-0.3, 0.4, 0.1499)
    goal_high = (-0.2, 0.5, 0.1501)
    moved_bodies = ("door",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("door")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        door = self.qpos_adr("doorjoint")
        d = sim.set_state(d, layout, d.qpos.at[door].set(0.), d.qvel.at[self.dof_adr("doorjoint")].set(0.))
        return d, dict(target_pos=rand_vec + jnp.array([-0.3, -0.45, 0.]), obj_init_pos=rand_vec)

    def pos_objects(self, d, episode):
        return self.geom_pos(d, "handle")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "handle")

    @staticmethod
    def _reward_pos(obs, theta):
        hand = obs[:3]
        door = obs[4:7] + jnp.array([-0.05, 0., 0.])
        threshold = 0.12
        radius = jnp.linalg.norm(hand[:2] - door[:2])
        floor = jnp.where(radius <= threshold, 0., 0.04 * jnp.log(jnp.maximum(radius - threshold, 1e-30)) + 0.4)
        above_floor = jnp.where(
            hand[2] >= floor, 1.,
            rewards.tolerance(floor - hand[2], bounds=(0., 0.01), margin=floor / 2., sigmoid="long_tail"),
        )
        in_place = rewards.tolerance(
            jnp.linalg.norm(hand - door - jnp.array([0.05, 0.03, -0.01])), bounds=(0., threshold / 2.), margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_open = rewards.hamacher_product(above_floor, in_place)
        door_angle = -theta
        opened = 0.2 * (theta < -np.pi / 90.) + 0.8 * rewards.tolerance(
            np.pi / 2. + np.pi / 6 - door_angle, bounds=(0., 0.5), margin=np.pi / 3., sigmoid="long_tail",
        )
        return ready_to_open, opened

    def evaluate(self, d, obs, action, episode):
        theta = d.qpos[self.qpos_adr("doorjoint")]
        reward_grab = (jnp.clip(action[3], -1., 1.) + 1.) / 2.
        ready, opened = self._reward_pos(obs, theta)
        reward = 2. * rewards.hamacher_product(ready, reward_grab) + 8. * opened
        success = jnp.abs(obs[4] - episode["target_pos"][0]) <= 0.08
        reward = jnp.where(success, 10., reward)
        return reward, info(
            success=success, near_object=ready, grasp_success=reward_grab >= 0.5, grasp_reward=reward_grab,
            in_place_reward=opened, obj_to_target=0., unscaled_reward=reward,
        )
