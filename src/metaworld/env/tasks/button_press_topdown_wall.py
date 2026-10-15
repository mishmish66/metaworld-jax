"""Meta-World's button-press-topdown-wall-v3: press a button from above, behind a wall."""
import jax.numpy as jnp

from metaworld.env.tasks.button_press_topdown import TopdownButtonTask


class ButtonPressTopdownWall(TopdownButtonTask):
    xml = "sawyer_button_press_topdown_wall.xml"

    def _tcp_closed(self, obs):
        return jnp.maximum(obs[3], 0.)
