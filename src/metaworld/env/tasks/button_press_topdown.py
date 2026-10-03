"""Meta-World's button-press-topdown-v3: press a button from above, and the base of its wall variant."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.button_press import ButtonTask


class TopdownButtonTask(ButtonTask):
    """A button pressed from above."""
    xml = "sawyer_button_press_topdown.xml"
    rand_low = (-0.1, 0.8, 0.115)
    rand_high = (0.1, 0.9, 0.115)
    button_offset = (0., 0., 0.193)
    axis = 2
    success_distance = 0.024
    near_button_radius = 0.01

    def _place(self, sim, d, layout):
        return sim.forward(d, layout)

    def _tcp_closed(self, obs):
        return 1. - obs[3]

    def _reward(self, obs, tcp_to_obj, near_button, button_pressed):
        reward = 5. * rewards.hamacher_product(self._tcp_closed(obs), near_button)
        return jnp.where(tcp_to_obj <= 0.03, reward + 5. * button_pressed, reward)


class ButtonPressTopdown(TopdownButtonTask):
    pass
