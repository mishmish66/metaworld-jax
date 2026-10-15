"""Meta-World's dial-turn-v3: turn a dial to a goal angle."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

# radius in meters of the dial's observed rim point
_DIAL_RADIUS = 0.05
# offset of the point the gripper pushes the dial at from its rim point
_PUSH_OFFSET = (0.05, 0.02, 0.09)


class DialTurn(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_dial.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.7, 0.0)
    rand_high = (0.1, 0.8, 0.0)
    goal_low = (-0.1, 0.73, 0.0299)
    goal_high = (0.1, 0.83, 0.0301)
    moved_bodies = ("dial",)
    TARGET_RADIUS = 0.07

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("dial")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        episode = dict(
            target_pos=rand_vec + jnp.array([0., 0.03, 0.03]), obj_init_pos=rand_vec, init_tcp=self.tcp_center(d),
            dial_push_position=self.pos_objects(d, None) + jnp.asarray(_PUSH_OFFSET),
        )
        return d, episode

    def pos_objects(self, d, episode):
        angle = d.qpos[self.qpos_adr("knob_Joint_1")]
        offset = jnp.stack([jnp.sin(angle), -jnp.cos(angle), 0.]) * _DIAL_RADIUS
        return self.body_com(d, "dial") + offset

    def quat_objects(self, d, episode):
        return d.xquat[self.body("dial")]

    def evaluate(self, d, obs, action, episode):
        obj = self.pos_objects(d, episode)
        dial_push_position = obj + jnp.asarray(_PUSH_OFFSET)
        tcp = self.tcp_center(d)
        target = episode["target_pos"]
        target_to_obj = jnp.linalg.norm(obj - target)
        target_to_obj_init = jnp.linalg.norm(episode["dial_push_position"] - target)
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=jnp.abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        dial_reach_radius = 0.005
        tcp_to_obj = jnp.linalg.norm(dial_push_position - tcp)
        tcp_to_obj_init = jnp.linalg.norm(episode["dial_push_position"] - episode["init_tcp"])
        reach = rewards.tolerance(
            tcp_to_obj, bounds=(0., dial_reach_radius), margin=jnp.abs(tcp_to_obj_init - dial_reach_radius),
            sigmoid="gaussian",
        )
        reach = rewards.hamacher_product(reach, jnp.minimum(jnp.maximum(0., action[-1]), 1.))
        reward = 10. * rewards.hamacher_product(reach, in_place)
        return reward, info(
            success=target_to_obj <= self.TARGET_RADIUS, near_object=tcp_to_obj <= 0.01, grasp_success=1.,
            grasp_reward=reach, in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )
