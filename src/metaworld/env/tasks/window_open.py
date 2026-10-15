"""Meta-World's window-open-v3: slide a window open."""
import jax.numpy as jnp

from metaworld.env import rewards
from metaworld.env.tasks.base import Task, info


class _Window(Task):
    """Sliding a window whose frame is placed at the random reset vector."""
    xml = "sawyer_window_horizontal.xml"
    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    hand_init_pos = (0., 0.4, 0.2)
    rand_low = (-0.1, 0.7, 0.16)
    rand_high = (0.1, 0.9, 0.16)
    goal_low = hand_low
    goal_high = hand_high
    moved_bodies = ("window",)
    handle_site: str
    # window slide position at reset
    slide_init_qpos: float
    reach_sigmoid: str

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("window")].set(rand_vec)

    def target(self, rand_vec):
        raise NotImplementedError

    def handle_init_pos(self, d):
        raise NotImplementedError

    def reset(self, sim, d, layout, rand_vec):
        episode = dict(
            target_pos=self.target(rand_vec), obj_init_pos=rand_vec, init_tcp=self.tcp_center(d),
            window_handle_pos_init=self.handle_init_pos(d),
        )
        # the slide is set without forwarding the kinematics
        return d.replace(qpos=d.qpos.at[self.qpos_adr("window_slide")].set(self.slide_init_qpos)), episode

    def pos_objects(self, d, episode):
        return self.site_pos(d, self.handle_site)

    def quat_objects(self, d, episode):
        return jnp.zeros(4)

    def target_to_obj_init(self, episode):
        raise NotImplementedError

    def evaluate(self, d, obs, action, episode):
        obj = self.pos_objects(d, episode)
        tcp = self.tcp_center(d)
        target = episode["target_pos"]
        target_to_obj = jnp.abs(obj[0] - target[0])
        in_place = rewards.tolerance(
            target_to_obj, bounds=(0., self.TARGET_RADIUS),
            margin=jnp.abs(self.target_to_obj_init(episode) - self.TARGET_RADIUS), sigmoid="long_tail",
        )
        handle_radius = 0.02
        tcp_to_obj = jnp.linalg.norm(obj - tcp)
        tcp_to_obj_init = jnp.linalg.norm(episode["window_handle_pos_init"] - episode["init_tcp"])
        reach = rewards.tolerance(
            tcp_to_obj, bounds=(0., handle_radius), margin=jnp.abs(tcp_to_obj_init - handle_radius),
            sigmoid=self.reach_sigmoid,
        )
        reward = 10. * rewards.hamacher_product(reach, in_place)
        return reward, info(
            success=target_to_obj <= self.TARGET_RADIUS, near_object=tcp_to_obj <= 0.05, grasp_success=1.,
            grasp_reward=reach, in_place_reward=in_place, obj_to_target=target_to_obj, unscaled_reward=reward,
        )


class WindowOpen(_Window):
    placed_sites = {"goal": "target_pos"}
    handle_site = "handleOpenStart"
    slide_init_qpos = 0.
    reach_sigmoid = "long_tail"

    def target(self, rand_vec):
        return rand_vec + jnp.array([0.2, 0., 0.])

    def handle_init_pos(self, d):
        return self.pos_objects(d, None)

    def target_to_obj_init(self, episode):
        return jnp.abs(episode["obj_init_pos"][0] - episode["target_pos"][0])
