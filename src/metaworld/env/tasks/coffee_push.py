"""Meta-World's coffee-push-v3: push a mug under a coffee machine."""
import jax.numpy as jnp

from metaworld.env.tasks.coffee_pull import MugTask


class CoffeePush(MugTask):
    physics = "upstream"
    placed_sites = {"mug_goal": "target_pos"}
    rand_low = (-0.1, 0.55, -0.001, -0.05, 0.7, -0.001)
    rand_high = (0.1, 0.65, 0.001, 0.05, 0.75, 0.001)
    goal_low = (-0.05, 0.7, -0.001)
    goal_high = (0.05, 0.75, 0.001)

    def _machine_pos(self, rand_vec):
        return rand_vec[3:] + jnp.array([0., 0.22, 0.])
