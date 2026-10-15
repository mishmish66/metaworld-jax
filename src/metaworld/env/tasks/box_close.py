"""Meta-World's box-close-v3: put the lid on a box."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class BoxClose(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_box.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.05, 0.5, 0.02, -0.1, 0.7, 0.133)
    rand_high = (0.05, 0.55, 0.02, 0.1, 0.8, 0.133)
    goal_low = (-0.1, 0.7, 0.133)
    goal_high = (0.1, 0.8, 0.133)
    moved_bodies = ("boxbody",)
    hand_reset_depends_on_layout = True

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.25)

    def layout(self, rand_vec, layout):
        i = self.layout_index("boxbody")
        return layout.at[i, :2].set(rand_vec[3:5])

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([0.02])])
        d = sim.step(d, layout, 5)
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, dict(target_pos=rand_vec[3:], obj_init_pos=obj_init_pos)

    def _main_object_geom(self):
        return self.geom("BoxHandleGeom")

    def pos_objects(self, d, episode):
        return self.body_com(d, "top_link")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("top_link")]

    @staticmethod
    def _reward_pos(obs, target_pos):
        hand = obs[:3]
        lid = obs[4:7] + jnp.array([0., 0., 0.02])
        threshold = 0.02
        radius = jnp.linalg.norm(hand[:2] - lid[:2])
        floor = jnp.where(radius <= threshold, 0., 0.04 * jnp.log(jnp.maximum(radius - threshold, 1e-30)) + 0.4)
        above_floor = jnp.where(
            hand[2] >= floor, 1.,
            rewards.tolerance(floor - hand[2], bounds=(0., 0.01), margin=floor / 2., sigmoid="long_tail"),
        )
        in_place = rewards.tolerance(jnp.linalg.norm(hand - lid), bounds=(0., 0.02), margin=0.5, sigmoid="long_tail")
        ready_to_lift = rewards.hamacher_product(above_floor, in_place)
        lifted = 0.2 * (lid[2] > 0.04) + 0.8 * rewards.tolerance(
            jnp.linalg.norm((target_pos - lid) * jnp.array([1., 1., 3.])), bounds=(0., 0.05), margin=0.25,
            sigmoid="long_tail",
        )
        return ready_to_lift, lifted

    def evaluate(self, d, obs, action, episode):
        target = episode["target_pos"]
        reward_grab = jnp.clip((jnp.clip(action[3], -1., 1.) + 1.) / 2., 0., 1.)
        quat = jnp.maximum(1. - jnp.linalg.norm(obs[7:11] - jnp.array([0.707, 0., 0., 0.707])) / 0.2, 0.)
        ready, lifted = self._reward_pos(obs, target)
        reward = 2. * rewards.hamacher_product(reward_grab, ready) + 8. * lifted
        success = jnp.linalg.norm(obs[4:7] - target) < 0.08
        reward = jnp.where(success, 10., reward) * quat
        return reward, info(
            success=success, near_object=ready, grasp_success=reward_grab >= 0.5, grasp_reward=reward_grab,
            in_place_reward=lifted, obj_to_target=0., unscaled_reward=reward,
        )
