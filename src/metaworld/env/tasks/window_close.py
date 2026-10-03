"""Meta-World's window-close-v3: slide a window closed."""
import jax.numpy as jnp

from metaworld.env.tasks.window_open import _Window


class WindowClose(_Window):
    placed_sites = {"goal": "target_pos"}
    rand_low = (0.0, 0.75, 0.2)
    rand_high = (0.0, 0.9, 0.2)
    handle_site = "handleCloseStart"
    slide_init_qpos = 0.2
    reach_sigmoid = "gaussian"

    def target(self, rand_vec):
        return rand_vec

    def handle_init_pos(self, d):
        return self.pos_objects(d, None) + jnp.array([0.2, 0., 0.])

    def target_to_obj_init(self, episode):
        return jnp.abs(episode["window_handle_pos_init"][0] - episode["target_pos"][0])
