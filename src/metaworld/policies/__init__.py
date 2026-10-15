"""
Meta-World's scripted policies as JAX functions from an observation to an action, by task name:

    from metaworld.policies import POLICIES
    action = POLICIES["pick-place"](obs)

The actions are unclipped, as Meta-World's are.
"""
from collections.abc import Callable

import jax

from metaworld.policies import assembly
from metaworld.policies import basketball
from metaworld.policies import bin_picking
from metaworld.policies import box_close
from metaworld.policies import button_press
from metaworld.policies import button_press_topdown
from metaworld.policies import button_press_topdown_wall
from metaworld.policies import button_press_wall
from metaworld.policies import coffee_button
from metaworld.policies import coffee_pull
from metaworld.policies import coffee_push
from metaworld.policies import dial_turn
from metaworld.policies import disassemble
from metaworld.policies import door_close
from metaworld.policies import door_lock
from metaworld.policies import door_open
from metaworld.policies import door_unlock
from metaworld.policies import drawer_close
from metaworld.policies import drawer_open
from metaworld.policies import faucet_close
from metaworld.policies import faucet_open
from metaworld.policies import hammer
from metaworld.policies import hand_insert
from metaworld.policies import handle_press
from metaworld.policies import handle_press_side
from metaworld.policies import handle_pull
from metaworld.policies import handle_pull_side
from metaworld.policies import lever_pull
from metaworld.policies import peg_insert_side
from metaworld.policies import peg_unplug_side
from metaworld.policies import pick_out_of_hole
from metaworld.policies import pick_place
from metaworld.policies import pick_place_wall
from metaworld.policies import plate_slide
from metaworld.policies import plate_slide_back
from metaworld.policies import plate_slide_back_side
from metaworld.policies import plate_slide_side
from metaworld.policies import push
from metaworld.policies import push_back
from metaworld.policies import push_wall
from metaworld.policies import reach
from metaworld.policies import reach_wall
from metaworld.policies import shelf_place
from metaworld.policies import soccer
from metaworld.policies import stick_pull
from metaworld.policies import stick_push
from metaworld.policies import sweep
from metaworld.policies import sweep_into
from metaworld.policies import window_close
from metaworld.policies import window_open

POLICIES: dict[str, Callable[[jax.Array], jax.Array]] = {
    "assembly": assembly.policy,
    "basketball": basketball.policy,
    "bin-picking": bin_picking.policy,
    "box-close": box_close.policy,
    "button-press": button_press.policy,
    "button-press-topdown": button_press_topdown.policy,
    "button-press-topdown-wall": button_press_topdown_wall.policy,
    "button-press-wall": button_press_wall.policy,
    "coffee-button": coffee_button.policy,
    "coffee-pull": coffee_pull.policy,
    "coffee-push": coffee_push.policy,
    "dial-turn": dial_turn.policy,
    "disassemble": disassemble.policy,
    "door-close": door_close.policy,
    "door-lock": door_lock.policy,
    "door-open": door_open.policy,
    "door-unlock": door_unlock.policy,
    "drawer-close": drawer_close.policy,
    "drawer-open": drawer_open.policy,
    "faucet-close": faucet_close.policy,
    "faucet-open": faucet_open.policy,
    "hammer": hammer.policy,
    "hand-insert": hand_insert.policy,
    "handle-press": handle_press.policy,
    "handle-press-side": handle_press_side.policy,
    "handle-pull": handle_pull.policy,
    "handle-pull-side": handle_pull_side.policy,
    "lever-pull": lever_pull.policy,
    "peg-insert-side": peg_insert_side.policy,
    "peg-unplug-side": peg_unplug_side.policy,
    "pick-out-of-hole": pick_out_of_hole.policy,
    "pick-place": pick_place.policy,
    "pick-place-wall": pick_place_wall.policy,
    "plate-slide": plate_slide.policy,
    "plate-slide-back": plate_slide_back.policy,
    "plate-slide-back-side": plate_slide_back_side.policy,
    "plate-slide-side": plate_slide_side.policy,
    "push": push.policy,
    "push-back": push_back.policy,
    "push-wall": push_wall.policy,
    "reach": reach.policy,
    "reach-wall": reach_wall.policy,
    "shelf-place": shelf_place.policy,
    "soccer": soccer.policy,
    "stick-pull": stick_pull.policy,
    "stick-push": stick_push.policy,
    "sweep": sweep.policy,
    "sweep-into": sweep_into.policy,
    "window-close": window_close.policy,
    "window-open": window_open.policy,
}

__all__ = ["POLICIES"]
