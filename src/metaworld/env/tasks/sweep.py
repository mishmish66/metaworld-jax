"""Meta-World's sweep-v3: sweep a puck off the table's edge."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


def centered_caging_reward(task, d, episode, action, obj_pos, obj_radius, grip_success_margin, x_z_success_margin):
    """
    The caging reward of Meta-World's soccer, sweep, and sweep-into: caging the object between the pads and
    over it in x-z, averaged with gripping it once caged.
    """
    pad_success_margin = 0.05
    tcp = task.tcp_center(d)
    left_pad, right_pad = task.body_com(d, "leftpad"), task.body_com(d, "rightpad")
    delta_object_y_left_pad = left_pad[1] - obj_pos[1]
    delta_object_y_right_pad = obj_pos[1] - right_pad[1]
    # Meta-World's initial pad positions are views of the current ones
    right_caging_margin = jnp.abs(jnp.abs(obj_pos[1] - right_pad[1]) - pad_success_margin)
    left_caging_margin = jnp.abs(jnp.abs(obj_pos[1] - left_pad[1]) - pad_success_margin)

    def tolerance(x, upper, margin):
        return rewards.tolerance(x, bounds=(obj_radius, upper), margin=margin, sigmoid="long_tail")

    y_caging = rewards.hamacher_product(
        tolerance(delta_object_y_right_pad, pad_success_margin, right_caging_margin),
        tolerance(delta_object_y_left_pad, pad_success_margin, left_caging_margin),
    )
    y_gripping = rewards.hamacher_product(
        tolerance(delta_object_y_right_pad, grip_success_margin, right_caging_margin),
        tolerance(delta_object_y_left_pad, grip_success_margin, left_caging_margin),
    )
    xz = jnp.array([1., 0., 1.])
    tcp_obj_norm_x_z = jnp.linalg.norm(tcp * xz - obj_pos * xz)
    tcp_obj_x_z_margin = jnp.linalg.norm(episode["obj_init_pos"] * xz - episode["init_tcp"] * xz) - x_z_success_margin
    x_z_caging = rewards.tolerance(
        tcp_obj_norm_x_z, bounds=(0., x_z_success_margin), margin=tcp_obj_x_z_margin, sigmoid="long_tail",
    )
    caging = rewards.hamacher_product(y_caging, x_z_caging)
    gripping = jnp.where(caging > 0.95, y_gripping, 0.)
    return (caging + gripping) / 2.


class Sweep(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_sweep_v3.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1.0, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.6, 0.02)
    rand_high = (0.1, 0.7, 0.02)
    goal_low = (0.49, 0.6, 0.00)
    goal_high = (0.51, 0.7, 0.02)
    OBJ_RADIUS = 0.02


    def reset(self, sim, d, layout, rand_vec):
        obj_init_pos = jnp.concatenate([rand_vec[:2], jnp.array([0.02])])
        target_pos = jnp.array([0.5, 0.65, 0.01]).at[1].set(rand_vec[1])
        episode = dict(target_pos=target_pos, obj_init_pos=obj_init_pos, init_tcp=self.tcp_center(d))
        d = sim.set_state(d, layout, d.qpos.at[9:12].set(obj_init_pos), d.qvel.at[9:15].set(0.))
        return d, episode

    def pos_objects(self, d, episode):
        return d.xpos[self.body("obj")]

    def quat_objects(self, d, episode):
        return d.xquat[self.body("obj")]

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        target = episode["target_pos"]
        obj_to_target = jnp.linalg.norm(obj - target)
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        in_place_margin = jnp.linalg.norm(episode["obj_init_pos"] - target)
        in_place = rewards.tolerance(obj_to_target, bounds=(0., 0.05), margin=in_place_margin, sigmoid="long_tail")
        object_grasped = centered_caging_reward(
            self, d, episode, action, obj, self.OBJ_RADIUS, self.OBJ_RADIUS + 0.01, 0.005,
        )
        reward = 2. * object_grasped + 6. * rewards.hamacher_product(object_grasped, in_place)
        reward = jnp.where(obj_to_target < 0.05, 10., reward)
        return reward, info(
            success=obj_to_target <= 0.05, near_object=tcp_to_obj <= 0.03,
            grasp_success=self.touching_main_object(d) & (obs[3] > 0), grasp_reward=object_grasped,
            in_place_reward=in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
