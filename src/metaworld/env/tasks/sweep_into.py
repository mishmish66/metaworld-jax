"""Meta-World's sweep-into-v3: sweep a puck into a hole in the table."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info
from metaworld.env.tasks.sweep import centered_caging_reward

_GOAL = (0., 0.84, 0.02)


class SweepInto(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_table_with_hole.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.6, 0.02, -0.001, 0.8399, 0.0199)
    rand_high = (0.1, 0.7, 0.02, 0.001, 0.8401, 0.0201)
    goal_low = (-0.001, 0.8399, 0.0199)
    goal_high = (0.001, 0.8401, 0.0201)
    OBJ_RADIUS = 0.02


    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - jnp.array(_GOAL[:2])) >= 0.15)

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], self.body_com(d, "obj")[2:]])
        episode = dict(target_pos=jnp.array(_GOAL), obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return self.body_com(d, "obj")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "objGeom")

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        target = jnp.concatenate([episode["target_pos"][:2], obj[2:]])
        obj_to_target = jnp.linalg.norm(obj - target)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        in_place_margin = jnp.linalg.norm(episode["obj_init_pos"] - target)
        in_place = rewards.tolerance(obj_to_target, bounds=(0., 0.05), margin=in_place_margin, sigmoid="long_tail")
        object_grasped = centered_caging_reward(
            self, d, episode, action, obj, self.OBJ_RADIUS, self.OBJ_RADIUS + 0.005, 0.01,
        )
        reward = 2. * object_grasped + 6. * rewards.hamacher_product(object_grasped, in_place)
        reward = jnp.where(obj_to_target < 0.05, 10., reward)
        return reward, info(
            success=obj_to_target <= 0.05, near_object=tcp_to_obj <= 0.03,
            grasp_success=self.touching_main_object(d) & (obs[3] > 0), grasp_reward=object_grasped,
            in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
