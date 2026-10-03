"""Meta-World's stick-push-v3: grasp a stick and push a box with it to a goal position."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

_STICK_Z = 0.02


class StickPush(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_stick_obj.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.08, 0.58, 0.000, 0.399, 0.55, 0.1319)
    rand_high = (-0.03, 0.62, 0.001, 0.401, 0.6, 0.1321)
    goal_low = (0.399, 0.55, 0.1319)
    goal_high = (0.401, 0.6, 0.1321)


    def sample(self, key):
        return self.sample_until(key, lambda v: jnp.linalg.norm(v[:2] - v[3:5]) >= 0.1)

    def reset(self, sim, d, layout, rand_vec):
        stick_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([_STICK_Z])])
        target_pos = jnp.concatenate([rand_vec[3:5], self.site_pos(d, "insertion")[2:]])
        init_tcp = self.tcp_center(d)
        qpos = d.qpos.at[9:12].set(stick_init_pos).at[16:18].set(0.)
        qvel = d.qvel.at[9:15].set(0.).at[16:18].set(0.)
        d = sim.set_state(d, layout, qpos, qvel)
        episode = dict(
            target_pos=target_pos, stick_init_pos=stick_init_pos, obj_init_pos=self.body_com(d, "object"),
            init_tcp=init_tcp,
        )
        return d, episode

    def pos_objects(self, d, episode):
        return jnp.concatenate([self.body_com(d, "stick"), self.site_pos(d, "insertion") + jnp.array([0., 0.09, 0.])])

    def quat_objects(self, d, episode):
        return jnp.concatenate([self.body_mat_quat(d, "stick"), jnp.zeros(4)])

    def evaluate(self, d, obs, action, episode):
        target_radius = 0.12
        tcp = self.tcp_center(d)
        stick = obs[4:7] + jnp.array([0.015, 0., 0.])
        container = obs[11:14]
        tcp_opened = obs[3]
        target, stick_init_pos = episode["target_pos"], episode["stick_init_pos"]
        tcp_to_stick = jnp.linalg.norm(stick - tcp)
        stick_to_target = jnp.linalg.norm(stick - target)
        stick_in_place_margin = jnp.linalg.norm(stick_init_pos - target) - target_radius
        stick_in_place = rewards.tolerance(
            stick_to_target, bounds=(0., target_radius), margin=stick_in_place_margin, sigmoid="long_tail",
        )
        container_to_target = jnp.linalg.norm(container - target)
        container_in_place_margin = jnp.linalg.norm(episode["obj_init_pos"] - target) - target_radius
        container_in_place = rewards.tolerance(
            container_to_target, bounds=(0., target_radius), margin=container_in_place_margin, sigmoid="long_tail",
        )
        # Meta-World cages the stick relative to its initial position
        object_grasped = self.gripper_caging_reward(
            d, dict(episode, obj_init_pos=stick_init_pos), action, stick, obj_radius=0.04, pad_success_thresh=0.05,
            object_reach_radius=0.01, xz_thresh=0.01, high_density=True,
        )
        grasped = (tcp_to_stick < 0.02) & (tcp_opened > 0) & (stick[2] - 0.01 > stick_init_pos[2])
        reward = jnp.where(
            grasped,
            jnp.where(
                container_to_target <= target_radius, 10., 2. + 5. * stick_in_place + 3. * container_in_place,
            ),
            object_grasped,
        )
        object_grasped = jnp.where(grasped, 1., object_grasped)
        grasp_success = self.touching_main_object(d) & (tcp_opened > 0) & (obs[6] - 0.01 > stick_init_pos[2])
        success = jnp.linalg.norm(container - target) <= 0.12
        return reward, info(
            success=grasp_success & success, near_object=tcp_to_stick <= 0.03, grasp_success=grasp_success,
            grasp_reward=object_grasped, in_place_reward=stick_in_place, obj_to_target=container_to_target,
            unscaled_reward=reward,
        )
