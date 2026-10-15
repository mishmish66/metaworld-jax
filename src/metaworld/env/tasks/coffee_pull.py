"""Meta-World's coffee-pull-v3: pull a mug from a coffee machine, and the base of coffee-push-v3."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class MugTask(Task):
    """A mug moved between a coffee machine and a goal."""
    xml = "sawyer_coffee.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (-0.05, 0.7, -0.001, -0.1, 0.55, -0.001)
    rand_high = (0.05, 0.75, 0.001, 0.1, 0.65, 0.001)
    goal_low = (-0.1, 0.55, -0.001)
    goal_high = (0.1, 0.65, 0.001)
    moved_bodies = ("coffee_machine",)

    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.15)

    def _machine_pos(self, rand_vec):
        return rand_vec[:3] + jnp.array([0., 0.22, 0.])

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("coffee_machine")].set(self._machine_pos(rand_vec))

    def reset(self, sim, d, layout, rand_vec):
        init_tcp = self.tcp_center(d)
        pos_mug_init, pos_mug_goal = rand_vec[:3], rand_vec[3:]
        d = sim.set_state(d, layout, d.qpos.at[0:3].set(pos_mug_init), d.qvel.at[9:15].set(0.))
        return d, dict(target_pos=pos_mug_goal, obj_init_pos=pos_mug_init, init_tcp=init_tcp)

    def _main_object_geom(self):
        return self.geom("mug")

    def pos_objects(self, d, episode):
        return self.body_com(d, "obj")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "mug")

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        target = episode["target_pos"]
        scale = jnp.array([2., 2., 1.])
        target_to_obj = jnp.linalg.norm((obj - target) * scale)
        target_to_obj_init = jnp.linalg.norm((episode["obj_init_pos"] - target) * scale)
        in_place = rewards.tolerance(target_to_obj, bounds=(0., 0.05), margin=target_to_obj_init, sigmoid="long_tail")
        tcp_opened = obs[3]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        object_grasped = self.gripper_caging_reward(
            d, episode, action, obj, object_reach_radius=0.04, obj_radius=0.02, pad_success_thresh=0.05,
            xz_thresh=0.05, desired_gripper_effort=0.7, medium_density=True,
        )
        reward = rewards.hamacher_product(object_grasped, in_place)
        reward = jnp.where((tcp_to_obj < 0.04) & (tcp_opened > 0), reward + 1. + 5. * in_place, reward)
        reward = jnp.where(target_to_obj < 0.05, 10., reward)
        obj_to_target = jnp.linalg.norm(obj - target)
        return reward, info(
            success=obj_to_target <= 0.07, near_object=tcp_to_obj <= 0.03,
            grasp_success=self.touching_main_object(d) & (tcp_opened > 0), grasp_reward=object_grasped,
            in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )


class CoffeePull(MugTask):
    placed_sites = {"mug_goal": "target_pos"}
