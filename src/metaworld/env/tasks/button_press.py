"""Meta-World's button-press-v3: press a button on a box facing the robot, and the base of its variants."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class ButtonTask(Task):
    """A button on a box the task places, pressed along one axis."""
    xml = "sawyer_button_press.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (-0.1, 0.85, 0.115)
    rand_high = (0.1, 0.9, 0.115)
    goal_low = hand_low
    goal_high = hand_high
    moved_bodies = ("box",)
    # offset of the observed button position from the button body
    button_offset = (0., -0.193, 0.)
    # axis the button moves along
    axis = 1
    success_distance = 0.02

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("box")].set(rand_vec)

    def _place(self, sim, d, layout):
        """World data after the button is placed, before the target is read."""
        return sim.set_state(d, layout, d.qpos.at[9].set(0.), d.qvel.at[9].set(0.))

    def reset(self, sim, d, layout, rand_vec):
        init_tcp = self.tcp_center(d)
        d = self._place(sim, d, layout)
        target_pos = self.site_pos(d, "hole")
        obj_to_target_init = jnp.abs(target_pos[self.axis] - self.site_pos(d, "buttonStart")[self.axis])
        return d, dict(target_pos=target_pos, obj_init_pos=rand_vec, init_tcp=init_tcp, obj_to_target_init=obj_to_target_init)

    def pos_objects(self, d, episode):
        return self.body_com(d, "button") + jnp.asarray(self.button_offset)

    def quat_objects(self, d, episode):
        return d.xquat[self.body("button")]

    def _distances(self, d, obs, episode):
        obj = obs[4:7]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        tcp_to_obj_init = jnp.linalg.norm(obj - episode["init_tcp"])
        obj_to_target = jnp.abs(episode["target_pos"][self.axis] - obj[self.axis])
        return tcp_to_obj, tcp_to_obj_init, obj_to_target

    def _reward(self, obs, tcp_to_obj, near_button, button_pressed):
        reward = 2. * rewards.hamacher_product(jnp.maximum(obs[3], 0.), near_button)
        return jnp.where(tcp_to_obj <= 0.05, reward + 8. * button_pressed, reward)

    near_button_radius = 0.05

    def evaluate(self, d, obs, action, episode):
        tcp_to_obj, tcp_to_obj_init, obj_to_target = self._distances(d, obs, episode)
        near_button = rewards.tolerance(
            tcp_to_obj, bounds=(0., self.near_button_radius), margin=tcp_to_obj_init, sigmoid="long_tail",
        )
        button_pressed = rewards.tolerance(
            obj_to_target, bounds=(0., 0.005), margin=episode["obj_to_target_init"], sigmoid="long_tail",
        )
        reward = self._reward(obs, tcp_to_obj, near_button, button_pressed)
        return reward, info(
            success=obj_to_target <= self.success_distance, near_object=tcp_to_obj <= 0.05, grasp_success=obs[3] > 0,
            grasp_reward=near_button, in_place_reward=button_pressed, obj_to_target=obj_to_target,
            unscaled_reward=reward,
        )


class ButtonPress(ButtonTask):
    pass
