"""Meta-World's plate-slide-v3: slide a puck into a goal net."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class PlateSlide(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_plate_slide.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (0.0, 0.6, 0.0, -0.1, 0.85, 0.0)
    rand_high = (0.0, 0.6, 0.0, 0.1, 0.9, 0.0)
    goal_low = (-0.1, 0.85, 0.0)
    goal_high = (0.1, 0.9, 0.0)
    moved_bodies = ("puck_goal",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("puck_goal")].set(rand_vec[3:])

    def reset(self, sim, d, layout, rand_vec):
        episode = dict(target_pos=rand_vec[3:], obj_init_pos=rand_vec[:3], init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:11].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.geom_pos(d, "puck")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "puck")

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        target, obj_init_pos = episode["target_pos"], episode["obj_init_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        in_place = rewards.tolerance(
            obj_to_target, bounds=(0., self.TARGET_RADIUS), margin=jnp.linalg.norm(obj_init_pos - target),
            sigmoid="long_tail",
        )
        tcp_to_obj = jnp.linalg.norm(tcp - obj)
        object_grasped = rewards.tolerance(
            tcp_to_obj, bounds=(0., self.TARGET_RADIUS), margin=jnp.linalg.norm(episode["init_tcp"] - obj_init_pos),
            sigmoid="long_tail",
        )
        reward = 8. * rewards.hamacher_product(object_grasped, in_place)
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10., reward)
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=0.,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
