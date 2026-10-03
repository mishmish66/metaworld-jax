"""Meta-World's coffee-button-v3: press the button of a coffee machine."""
import numpy as np

import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

# distance in meters the button travels
_MAX_DIST = 0.03


class CoffeeButton(Task):
    xml = "sawyer_coffee.xml"
    hand_low = (-0.5, 0.4, 0.05)
    hand_high = (0.5, 1.0, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (-0.1, 0.8, -0.001)
    rand_high = (0.1, 0.9, 0.001)
    goal_low = tuple(np.array(rand_low) + np.array([-0.001, -0.22 + _MAX_DIST, 0.299]))
    goal_high = tuple(np.array(rand_high) + np.array([0.001, -0.22 + _MAX_DIST, 0.301]))
    moved_bodies = ("coffee_machine",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("coffee_machine")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        init_tcp = self.tcp_center(d)
        pos_mug = rand_vec + jnp.array([0., -0.22, 0.])
        d = sim.set_state(d, layout, d.qpos.at[0:3].set(pos_mug), d.qvel.at[9:15].set(0.))
        target_pos = rand_vec + jnp.array([0., -0.22 + _MAX_DIST, 0.3])
        return d, dict(target_pos=target_pos, obj_init_pos=rand_vec, init_tcp=init_tcp)

    def pos_objects(self, d, episode):
        return self.site_pos(d, "buttonStart")

    def quat_objects(self, d, episode):
        return jnp.array([1., 0., 0., 0.])

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        tcp_to_obj = jnp.linalg.norm(obj - self.tcp_center(d))
        tcp_to_obj_init = jnp.linalg.norm(obj - episode["init_tcp"])
        obj_to_target = jnp.abs(episode["target_pos"][1] - obj[1])
        near_button = rewards.tolerance(tcp_to_obj, bounds=(0., 0.05), margin=tcp_to_obj_init, sigmoid="long_tail")
        button_pressed = rewards.tolerance(obj_to_target, bounds=(0., 0.005), margin=_MAX_DIST, sigmoid="long_tail")
        reward = 2. * rewards.hamacher_product(jnp.maximum(obs[3], 0.), near_button)
        reward = jnp.where(tcp_to_obj <= 0.05, reward + 8. * button_pressed, reward)
        return reward, info(
            success=obj_to_target <= 0.02, near_object=tcp_to_obj <= 0.05, grasp_success=obs[3] > 0,
            grasp_reward=near_button, in_place_reward=button_pressed, obj_to_target=obj_to_target,
            unscaled_reward=reward,
        )
