"""Meta-World's plate-slide-back-v3: slide a puck out of a goal net, back toward the robot."""
from metaworld.env.tasks.plate_slide_side import PlateSlideSide


class PlateSlideBack(PlateSlideSide):
    placed_sites = {"goal": "target_pos"}
    xml = "sawyer_plate_slide.xml"
    rand_low = (0.0, 0.85, 0.0, -0.1, 0.6, 0.015)
    rand_high = (0.0, 0.85, 0.0, 0.1, 0.6, 0.015)
    goal_low = (-0.1, 0.6, 0.015)
    goal_high = (0.1, 0.6, 0.015)
    puck_qpos = (0., 0.15)
