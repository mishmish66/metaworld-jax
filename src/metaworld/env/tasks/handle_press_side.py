"""Meta-World's handle-press-side-v3: press down a handle that faces sideways."""
from metaworld.env.tasks.handle_press import _HandlePress


class HandlePressSide(_HandlePress):
    xml = "sawyer_handle_press_sideways.xml"
    rand_low = (-0.35, 0.65, -0.001)
    rand_high = (-0.25, 0.75, 0.001)
    goal_low = _HandlePress.hand_low
    goal_high = _HandlePress.hand_high
