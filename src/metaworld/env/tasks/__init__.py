"""Meta-World's tasks, by name."""
from metaworld.env.tasks.base import INFO_KEYS, Task
from metaworld.env.tasks.assembly import Assembly
from metaworld.env.tasks.basketball import Basketball
from metaworld.env.tasks.bin_picking import BinPicking
from metaworld.env.tasks.box_close import BoxClose
from metaworld.env.tasks.button_press import ButtonPress
from metaworld.env.tasks.button_press_topdown import ButtonPressTopdown
from metaworld.env.tasks.button_press_topdown_wall import ButtonPressTopdownWall
from metaworld.env.tasks.button_press_wall import ButtonPressWall
from metaworld.env.tasks.coffee_button import CoffeeButton
from metaworld.env.tasks.coffee_pull import CoffeePull
from metaworld.env.tasks.coffee_push import CoffeePush
from metaworld.env.tasks.dial_turn import DialTurn
from metaworld.env.tasks.disassemble import Disassemble
from metaworld.env.tasks.door_close import DoorClose
from metaworld.env.tasks.door_lock import DoorLock
from metaworld.env.tasks.door_open import DoorOpen
from metaworld.env.tasks.door_unlock import DoorUnlock
from metaworld.env.tasks.drawer_close import DrawerClose
from metaworld.env.tasks.drawer_open import DrawerOpen
from metaworld.env.tasks.faucet_close import FaucetClose
from metaworld.env.tasks.faucet_open import FaucetOpen
from metaworld.env.tasks.hammer import Hammer
from metaworld.env.tasks.hand_insert import HandInsert
from metaworld.env.tasks.handle_press import HandlePress
from metaworld.env.tasks.handle_press_side import HandlePressSide
from metaworld.env.tasks.handle_pull import HandlePull
from metaworld.env.tasks.handle_pull_side import HandlePullSide
from metaworld.env.tasks.lever_pull import LeverPull
from metaworld.env.tasks.peg_insert_side import PegInsertSide
from metaworld.env.tasks.peg_unplug_side import PegUnplugSide
from metaworld.env.tasks.pick_out_of_hole import PickOutOfHole
from metaworld.env.tasks.pick_place import PickPlace
from metaworld.env.tasks.pick_place_wall import PickPlaceWall
from metaworld.env.tasks.plate_slide import PlateSlide
from metaworld.env.tasks.plate_slide_back import PlateSlideBack
from metaworld.env.tasks.plate_slide_back_side import PlateSlideBackSide
from metaworld.env.tasks.plate_slide_side import PlateSlideSide
from metaworld.env.tasks.push import Push
from metaworld.env.tasks.push_back import PushBack
from metaworld.env.tasks.push_wall import PushWall
from metaworld.env.tasks.reach import Reach
from metaworld.env.tasks.reach_wall import ReachWall
from metaworld.env.tasks.shelf_place import ShelfPlace
from metaworld.env.tasks.soccer import Soccer
from metaworld.env.tasks.stick_pull import StickPull
from metaworld.env.tasks.stick_push import StickPush
from metaworld.env.tasks.sweep import Sweep
from metaworld.env.tasks.sweep_into import SweepInto
from metaworld.env.tasks.window_close import WindowClose
from metaworld.env.tasks.window_open import WindowOpen

TASKS: dict[str, type[Task]] = {
    "assembly": Assembly,
    "basketball": Basketball,
    "bin-picking": BinPicking,
    "box-close": BoxClose,
    "button-press": ButtonPress,
    "button-press-topdown": ButtonPressTopdown,
    "button-press-topdown-wall": ButtonPressTopdownWall,
    "button-press-wall": ButtonPressWall,
    "coffee-button": CoffeeButton,
    "coffee-pull": CoffeePull,
    "coffee-push": CoffeePush,
    "dial-turn": DialTurn,
    "disassemble": Disassemble,
    "door-close": DoorClose,
    "door-lock": DoorLock,
    "door-open": DoorOpen,
    "door-unlock": DoorUnlock,
    "drawer-close": DrawerClose,
    "drawer-open": DrawerOpen,
    "faucet-close": FaucetClose,
    "faucet-open": FaucetOpen,
    "hammer": Hammer,
    "hand-insert": HandInsert,
    "handle-press": HandlePress,
    "handle-press-side": HandlePressSide,
    "handle-pull": HandlePull,
    "handle-pull-side": HandlePullSide,
    "lever-pull": LeverPull,
    "peg-insert-side": PegInsertSide,
    "peg-unplug-side": PegUnplugSide,
    "pick-out-of-hole": PickOutOfHole,
    "pick-place": PickPlace,
    "pick-place-wall": PickPlaceWall,
    "plate-slide": PlateSlide,
    "plate-slide-back": PlateSlideBack,
    "plate-slide-back-side": PlateSlideBackSide,
    "plate-slide-side": PlateSlideSide,
    "push": Push,
    "push-back": PushBack,
    "push-wall": PushWall,
    "reach": Reach,
    "reach-wall": ReachWall,
    "shelf-place": ShelfPlace,
    "soccer": Soccer,
    "stick-pull": StickPull,
    "stick-push": StickPush,
    "sweep": Sweep,
    "sweep-into": SweepInto,
    "window-close": WindowClose,
    "window-open": WindowOpen,
}

__all__ = ["INFO_KEYS", "TASKS", "Task"]
