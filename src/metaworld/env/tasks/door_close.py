"""Meta-World's door-close-v3: close an open door with a revolving joint."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class DoorClose(Task):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_door_pull.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (-0.5, 0.6, 0.2)
    rand_low = (0.0, 0.85, 0.15)
    rand_high = (0.1, 0.95, 0.15)
    goal_low = (0.2, 0.65, 0.1499)
    goal_high = (0.3, 0.75, 0.1501)
    moved_bodies = ("door",)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("door")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        # the door starts open
        door = self.qpos_adr("doorjoint")
        d = sim.set_state(d, layout, d.qpos.at[door].set(-1.5708), d.qvel.at[self.dof_adr("doorjoint")].set(0.))
        return d, dict(target_pos=rand_vec + jnp.array([0.2, -0.2, 0.]), obj_init_pos=rand_vec)

    def pos_objects(self, d, episode):
        return self.geom_pos(d, "handle")

    def quat_objects(self, d, episode):
        return self.geom_quat(d, "handle")

    def evaluate(self, d, obs, action, episode):
        tcp = self.tcp_center(d)
        obj = obs[4:7]
        target = episode["target_pos"]
        tcp_to_target = jnp.linalg.norm(tcp - target)
        obj_to_target = jnp.linalg.norm(obj - target)
        in_place_margin = jnp.linalg.norm(episode["obj_init_pos"] - target)
        in_place = rewards.tolerance(obj_to_target, bounds=(0., self.TARGET_RADIUS), margin=in_place_margin, sigmoid="gaussian")
        hand_margin = jnp.linalg.norm(jnp.asarray(self.hand_init_pos) - obj) + 0.1
        hand_in_place = rewards.tolerance(
            tcp_to_target, bounds=(0., 0.25 * self.TARGET_RADIUS), margin=hand_margin, sigmoid="gaussian",
        )
        reward = 3. * hand_in_place + 6. * in_place
        reward = jnp.where(obj_to_target < self.TARGET_RADIUS, 10., reward)
        return reward, info(
            success=obj_to_target <= 0.08, near_object=0., grasp_success=1., grasp_reward=1.,
            in_place_reward=hand_in_place, obj_to_target=obj_to_target, unscaled_reward=reward,
        )
