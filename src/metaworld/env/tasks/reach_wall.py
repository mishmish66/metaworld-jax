"""Meta-World's reach-wall-v3: move the gripper to a goal position behind a wall."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class ReachWall(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_reach_wall_v3.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.05, 0.6, 0.015, -0.05, 0.85, 0.05)
    rand_high = (0.05, 0.65, 0.015, 0.05, 0.9, 0.3)
    goal_low = (-0.05, 0.85, 0.05)
    goal_high = (0.05, 0.9, 0.3)

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.15)

    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos, target_pos = rand_vec[:3], rand_vec[3:]
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, dict(target_pos=target_pos, obj_init_pos=obj_init_pos)

    def pos_objects(self, d, episode):
        return self.body_com(d, "obj")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "objGeom")

    def evaluate(self, d, obs, action, episode):
        target = episode["target_pos"]
        tcp_to_target = jnp.linalg.norm(self.tcp_center(d) - target)
        in_place_margin = jnp.linalg.norm(jnp.asarray(self.hand_init_pos) - target)
        in_place = rewards.tolerance(tcp_to_target, bounds=(0., 0.05), margin=in_place_margin, sigmoid="long_tail")
        reward = 10. * in_place
        return reward, info(
            success=tcp_to_target <= 0.05, near_object=0., grasp_success=0., grasp_reward=0.,
            in_place_reward=in_place, obj_to_target=tcp_to_target, unscaled_reward=reward,
        )
