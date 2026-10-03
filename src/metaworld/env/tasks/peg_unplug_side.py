"""Meta-World's peg-unplug-side-v3: pull a plug sideways out of a socket."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class PegUnplugSide(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_peg_unplug_side.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.25, 0.6, -0.001)
    rand_high = (-0.15, 0.8, 0.001)
    goal_low = (-0.056, 0.6, 0.13)
    goal_high = (0.044, 0.8, 0.132)
    moved_bodies = ("box",)
    hand_reset_depends_on_layout = True

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("box")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        pos_plug = rand_vec + jnp.array([0.044, 0., 0.131])
        init_tcp = self.tcp_center(d)
        qpos = d.qpos.at[9:12].set(pos_plug).at[12:16].set(jnp.array([1., 0., 0., 0.]))
        d = sim.set_state(d, layout, qpos, d.qvel.at[9:12].set(0.))
        episode = dict(
            target_pos=pos_plug + jnp.array([0.15, 0., 0.]), obj_init_pos=self.site_pos(d, "pegEnd"),
            init_tcp=init_tcp,
        )
        return d, episode

    def pos_objects(self, d, episode):
        return self.site_pos(d, "pegEnd")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("plug1")]

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = episode["target_pos"]
        obj_init_pos = episode["obj_init_pos"]
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        obj_to_target = jnp.linalg.norm(obj - target)
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, object_reach_radius=0.01, obj_radius=0.025, pad_success_thresh=0.05,
            xz_thresh=0.005, desired_gripper_effort=0.8, high_density=True,
        )
        in_place = rewards.tolerance(
            obj_to_target, bounds=(0., 0.05), margin=jnp.linalg.norm(obj_init_pos - target), sigmoid="long_tail",
        )
        grasp_success = (tcp_opened > 0.5) & (obj[0] - obj_init_pos[0] > 0.015)
        reward = jnp.where(grasp_success & (tcp_to_obj < 0.035), 1. + 2. * object_grasped + 5. * in_place, 2. * object_grasped)
        reward = jnp.where(obj_to_target <= 0.05, 10., reward)
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
