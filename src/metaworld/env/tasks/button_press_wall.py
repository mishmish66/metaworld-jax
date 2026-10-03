"""Meta-World's button-press-wall-v3: press a button behind a wall."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.button_press import ButtonTask


class ButtonPressWall(ButtonTask):
    xml = "sawyer_button_press_wall.xml"
    rand_low = (-0.05, 0.85, 0.1149)
    rand_high = (0.05, 0.9, 0.1151)
    success_distance = 0.03
    near_button_radius = 0.01

    def _reward(self, obs, tcp_to_obj, near_button, button_pressed):
        far = 2. * rewards.hamacher_product((1. - obs[3]) / 2., near_button)
        near = 2. + 2. * (1. + obs[3]) + 4. * button_pressed ** 2
        return jnp.where(tcp_to_obj > 0.07, far, near)
