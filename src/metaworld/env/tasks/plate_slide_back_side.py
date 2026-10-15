"""Meta-World's plate-slide-back-side-v3: slide a puck sideways out of a goal net."""
from metaworld.env.tasks.plate_slide_side import PlateSlideSide


class PlateSlideBackSide(PlateSlideSide):
    placed_sites = {"goal": "target_pos"}
    rand_low = (-0.25, 0.6, 0.0, -0.05, 0.6, 0.015)
    rand_high = (-0.25, 0.6, 0.0, 0.15, 0.6, 0.015)
    goal_low = (-0.05, 0.6, 0.015)
    goal_high = (0.15, 0.6, 0.015)
    moved_bodies = ("puck_goal",)
    puck_qpos = (-0.15, 0.)

    def layout(self, rand_vec, layout):
        return layout.at[self.layout_index("puck_goal")].set(rand_vec[:3])
