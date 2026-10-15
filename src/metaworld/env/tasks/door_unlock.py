"""Meta-World's door-unlock-v3: turn a door's lock sideways."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info

LOCK_LENGTH = 0.1
# offset from the fingers to the gripper's shoulder
_SHOULDER = jnp.array([0., 0.055, 0.07])


class DoorUnlock(Task):
    xml = "sawyer_door_lock.xml"
    hand_low = (-0.5, 0.40, -0.15)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.8, 0.15)
    rand_high = (0.1, 0.85, 0.15)
    goal_low = (0.0, 0.64, 0.2100)
    goal_high = (0.2, 0.7, 0.2111)
    moved_bodies = ("door",)
    hand_reset_depends_on_layout = True

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("door")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        init_tcp = self.tcp_center(d)
        d = sim.set_state(d, layout, d.qpos.at[9].set(1.5708), d.qvel.at[9].set(0.))
        target_pos = d.xpos[self.body("lock_link")] + jnp.array([0.1, -0.04, 0.])
        return d, dict(target_pos=target_pos, init_tcp=init_tcp)

    def pos_objects(self, d, episode):
        return self.site_pos(d, "lockStartUnlock")

    def quat_objects(self, d, episode):
        return d.xquat[self.body("door_link")]

    def evaluate(self, d, obs, action, episode):
        gripper = obs[:3]
        lock = obs[4:7]
        scale = jnp.array([0.25, 1., 0.5])
        shoulder_to_lock = jnp.linalg.norm((gripper + _SHOULDER - lock) * scale)
        # Meta-World's initial lock position is a view of the current one
        shoulder_to_lock_init = (episode["init_tcp"] + _SHOULDER - d.xpos[self.body("lock_link")]) * scale
        ready_to_push = rewards.tolerance(
            shoulder_to_lock, bounds=(0., 0.02), margin=jnp.linalg.norm(shoulder_to_lock_init), sigmoid="long_tail",
        )
        obj_to_target = jnp.abs(episode["target_pos"][0] - lock[0])
        pushed = rewards.tolerance(obj_to_target, bounds=(0., 0.005), margin=LOCK_LENGTH, sigmoid="long_tail")
        reward = 2. * ready_to_push + 8. * pushed
        return reward, info(
            success=obj_to_target <= 0.02, near_object=shoulder_to_lock <= 0.05, grasp_success=obs[3] > 0,
            grasp_reward=ready_to_push, in_place_reward=pushed, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
