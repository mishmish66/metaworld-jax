"""Meta-World's stick-pull-v3: grasp a stick, insert it into a box's handle, and pull the box to a goal position."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

_STICK_Z = 0.02


class StickPull(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_stick_obj.xml"
    physics = "upstream"
    hand_low = (-0.5, 0.35, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.55, 0.000, 0.35, 0.45, 0.0199)
    rand_high = (0.0, 0.65, 0.001, 0.45, 0.55, 0.0201)
    goal_low = (0.35, 0.45, 0.0199)
    goal_high = (0.45, 0.55, 0.0201)


    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.1)

    def reset(self, sim, d, layout, rand_vec):
        stick_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([_STICK_Z])])
        target_pos = jnp.concatenate([rand_vec[3:5], jnp.array([_STICK_Z])])
        init_tcp = self.tcp_center(d)
        qpos = d.qpos.at[9:12].set(stick_init_pos).at[16:18].set(jnp.array([0., 0.09]))
        qvel = d.qvel.at[9:15].set(0.).at[16:18].set(0.)
        d = sim.set_state(d, layout, qpos, qvel)
        episode = dict(
            target_pos=target_pos, stick_init_pos=stick_init_pos, obj_init_pos=self.body_com(d, "object"),
            init_tcp=init_tcp,
        )
        return d, episode

    def pos_objects(self, d, episode):
        return jnp.concatenate([self.body_com(d, "stick"), self.site_pos(d, "insertion")])

    def quat_objects(self, d, episode):
        return jnp.concatenate([self.body_mat_quat(d, "stick"), jnp.zeros(4)])

    @staticmethod
    def _stick_is_inserted(handle, end_of_stick):
        return (
            (end_of_stick[0] >= handle[0]) & (jnp.abs(end_of_stick[1] - handle[1]) <= 0.040)
            & (jnp.abs(end_of_stick[2] - handle[2]) <= 0.060)
        )

    def evaluate(self, d, obs, action, episode):
        target_radius = 0.05
        tcp = self.tcp_center(d)
        stick = obs[4:7]
        end_of_stick = self.site_pos(d, "stick_end")
        container = obs[11:14] + jnp.array([0.05, 0., 0.])
        obj_init_pos, stick_init_pos = episode["obj_init_pos"], episode["stick_init_pos"]
        container_init_pos = obj_init_pos + jnp.array([0.05, 0., 0.])
        handle = obs[11:14]
        tcp_opened = obs[3]
        target = episode["target_pos"]
        tcp_to_stick = jnp.linalg.norm(stick - tcp)
        handle_to_target = jnp.linalg.norm(handle - target)
        yz_scaling = jnp.array([1., 1., 2.])
        stick_to_container = jnp.linalg.norm((stick - container) * yz_scaling)
        stick_in_place_margin = jnp.linalg.norm((stick_init_pos - container_init_pos) * yz_scaling)
        stick_in_place = rewards.tolerance(
            stick_to_container, bounds=(0., target_radius), margin=stick_in_place_margin, sigmoid="long_tail",
        )
        stick_to_target = jnp.linalg.norm(stick - target)
        stick_in_place_2 = rewards.tolerance(
            stick_to_target, bounds=(0., target_radius), margin=jnp.linalg.norm(stick_init_pos - target),
            sigmoid="long_tail",
        )
        container_to_target = jnp.linalg.norm(container - target)
        container_in_place = rewards.tolerance(
            container_to_target, bounds=(0., target_radius), margin=jnp.linalg.norm(obj_init_pos - target),
            sigmoid="long_tail",
        )
        object_grasped = self.gripper_caging_reward(
            d, episode, action, stick, obj_radius=0.014, pad_success_thresh=0.05, object_reach_radius=0.01,
            xz_thresh=0.01, high_density=True,
        )
        grasped = (tcp_to_stick < 0.02) & (tcp_opened > 0) & (stick[2] - 0.01 > stick_init_pos[2])
        object_grasped = jnp.where(grasped, 1., object_grasped)
        in_place_and_object_grasped = rewards.hamacher_product(object_grasped, stick_in_place)
        inserted = self._stick_is_inserted(handle, end_of_stick)
        reward = jnp.where(
            grasped,
            jnp.where(
                inserted,
                jnp.where(
                    handle_to_target <= 0.12, 10.,
                    1. + in_place_and_object_grasped + 5. + 2. * stick_in_place_2 + container_in_place,
                ),
                1. + in_place_and_object_grasped + 5. * stick_in_place,
            ),
            in_place_and_object_grasped,
        )
        grasp_success = self.touching_main_object(d) & (tcp_opened > 0) & (stick[2] - 0.02 > obj_init_pos[2])
        return reward, info(
            success=(handle_to_target <= 0.12) & inserted, near_object=tcp_to_stick <= 0.03,
            grasp_success=grasp_success, grasp_reward=object_grasped, in_place_reward=stick_in_place,
            obj_to_target=handle_to_target, unscaled_reward=reward,
        )
