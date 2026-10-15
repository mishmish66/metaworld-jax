"""Meta-World's handle-press-v3: press a handle down."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class _HandlePress(Task):
    """Pressing down a handle on a box placed at the random reset vector."""
    xml = "sawyer_handle_press.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1.0, 0.5)
    hand_init_pos = (0., 0.6, 0.2)
    rand_low = (-0.1, 0.8, -0.001)
    rand_high = (0.1, 0.9, 0.001)
    goal_low = (-0.1, 0.55, 0.04)
    goal_high = (0.1, 0.70, 0.08)
    moved_bodies = ("box",)
    TARGET_RADIUS = 0.02

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("box")].set(rand_vec)

    def reset(self, sim, d, layout, rand_vec):
        init_tcp = self.tcp_center(d)
        d = sim.set_state(d, layout, d.qpos.at[9].set(-0.001), d.qvel.at[9].set(0.))
        episode = dict(
            target_pos=self.site_pos(d, "goalPress"), obj_init_pos=rand_vec, init_tcp=init_tcp,
            handle_init_pos=self.pos_objects(d, None),
        )
        return d, episode

    def pos_objects(self, d, episode):
        return self.site_pos(d, "handleStart")

    def quat_objects(self, d, episode):
        return jnp.zeros(4)

    def evaluate(self, d, obs, action, episode):
        obj = self.pos_objects(d, episode)
        tcp = self.tcp_center(d)
        target = episode["target_pos"]
        handle_init_pos = episode["handle_init_pos"]
        target_to_obj = jnp.abs(obj[2] - target[2])
        target_to_obj_init = jnp.abs(handle_init_pos[2] - target[2])
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS), margin=jnp.abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )
        handle_radius = 0.02
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        tcp_to_obj_init = jnp.linalg.norm(handle_init_pos - episode["init_tcp"])
        reach = rewards.tolerance(
            tcp_to_obj, bounds=(0., handle_radius), margin=jnp.abs(tcp_to_obj_init - handle_radius), sigmoid="long_tail",
        )
        reward = rewards.hamacher_product(reach, in_place)
        reward = 10. * jnp.where(target_to_obj <= self.TARGET_RADIUS, 1., reward)
        return reward, info(
            success=target_to_obj <= self.TARGET_RADIUS, near_object=tcp_to_obj <= 0.05, grasp_success=1.,
            grasp_reward=reach, in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )


class HandlePress(_HandlePress):
    hand_reset_depends_on_layout = True
