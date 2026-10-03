"""Meta-World's door-lock-v3: turn a door's lock down."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

LOCK_LENGTH = 0.1


class DoorLock(Task):
    xml = "sawyer_door_lock.xml"
    hand_low = (-0.5, 0.40, -0.15)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.8, 0.15)
    rand_high = (0.1, 0.85, 0.15)
    goal_low = hand_low
    goal_high = hand_high
    moved_bodies = ("door",)
    hand_reset_depends_on_layout = True

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("door")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        # one policy step of simulation with the door in place
        d = sim.step(d, layout, sim.substeps)
        return d, dict(target_pos=d.xpos[self.body("lock_link")] + jnp.array([0., -0.04, -0.1]))

    def pos_objects(self, d, episode):
        return self.site_pos(d, "lockStartLock")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("door_link")]

    def evaluate(self, d, obs, action, episode):
        obj = obs[4:7]
        tcp = self.body_com(d, "leftpad")
        scale = jnp.array([0.25, 1., 0.5])
        tcp_to_obj = jnp.linalg.norm((obj - tcp) * scale)
        # Meta-World's initial left pad position is a view of the current one
        tcp_to_obj_init = tcp_to_obj
        obj_to_target = jnp.abs(episode["target_pos"][2] - obj[2])
        tcp_opened = jnp.maximum(obs[3], 0.)
        near_lock = rewards.tolerance(tcp_to_obj, bounds=(0., 0.01), margin=tcp_to_obj_init, sigmoid="long_tail")
        lock_pressed = rewards.tolerance(obj_to_target, bounds=(0., 0.005), margin=LOCK_LENGTH, sigmoid="long_tail")
        reward = 2. * rewards.hamacher_product(tcp_opened, near_lock) + 8. * lock_pressed
        return reward, info(
            success=obj_to_target <= 0.02, near_object=tcp_to_obj <= 0.05, grasp_success=obs[3] > 0,
            grasp_reward=near_lock, in_place_reward=lock_pressed, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
