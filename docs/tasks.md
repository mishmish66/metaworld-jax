# Tasks

Meta-World's 50 v3 tasks as POMDPs, made with `jax_pomdps.make("metaworld.env:<name>")`, named as in Meta-World
without "-v3". Every task is Meta-World's: its scene, reset distribution, observation, reward, and success check,
re-expressed in JAX on MuJoCo Warp. The descriptions are Meta-World's; the images show the first state of an episode
and its state when Meta-World's scripted policy succeeds.

Each task is registered in each observation mode as `name/<mode>`, and as `name` in the default mode, `mkv`:

- `name/mkv`, observing Meta-World's 39-vector with the goal: close to a Markov state, enough for an agent that sees
  one observation at a time;
- `name/pix`, observing `corner2_image`, an 84×84 uint8 image of the `corner2` camera, which shows the goal as
  Meta-World's images do, with its site marker;
- `name/pix-prp`, observing that image and `proprio`, the hand position and gripper opening at this step and the
  previous one.

Proprioception alone shows neither the objects nor the goal, so there is no `prp` mode. Meta-World's ML benchmarks
hide the goal, which an agent can only infer from rewards over episodes; `goal_observable=False` gives that
observation, which is not a registered mode.

`camera_names`, `camera_height`, and `camera_width` change the cameras, and every other argument of
`MetaworldPOMDP` (such as `max_worlds` or `physics`) passes through.

All tasks share these conventions:

- **Actions** are 4 values in [-1, 1]: a displacement of the hand's mocap target of up to 1 cm per axis, and the
  gripper's effort, positive closing it.
- **Control** runs at 80 Hz, Meta-World's rate. Each step is 3 substeps of 4.17 ms by default (`physics="fast"`),
  or Meta-World's 5 substeps of 2.5 ms with `physics="upstream"`.
- **Observations** are Meta-World's 39-vector: the hand position, the gripper opening, the positions and quaternions
  of up to two objects (padded with zeros to 14 values), the same 18 values at the previous step, and the goal
  position. Quaternions are as Meta-World observes them, (x, y, z, w) or MuJoCo's (w, x, y, z) depending on the task.
- **Reward** is Meta-World's v2 reward, at most 10; `evaluate` also returns Meta-World's info dict.
- **Episodes** never end on their own unless `terminate_on_success=True`; Meta-World's horizon is
  `MAX_PATH_LENGTH` = 500 steps, left to the caller.
- **Initial states** follow Meta-World's reset: the hand moves to its initial position, then the task places its
  objects and goal from a random reset vector drawn from the task's reset space.

## `assembly`

<img src="images/assembly.png" alt="Assemble nut: the first state and the state at success" loading="lazy">

**Assemble nut.** Pick up a nut and place it onto a peg. Randomize nut and peg positions.

## `basketball`

<img src="images/basketball.png" alt="Basketball: the first state and the state at success" loading="lazy">

**Basketball.** Dunk the basketball into the basket. Randomize basketball and basket positions.

## `bin-picking`

<img src="images/bin-picking.png" alt="Pick bin: the first state and the state at success" loading="lazy">

**Pick bin.** Grasp the puck from one bin and place it into another bin. Randomize puck positions.

## `box-close`

<img src="images/box-close.png" alt="Close box: the first state and the state at success" loading="lazy">

**Close box.** Grasp the cover and close the box with it. Randomize the cover and box positions.

## `button-press`

<img src="images/button-press.png" alt="Press button: the first state and the state at success" loading="lazy">

**Press button.** Press a button. Randomize button positions.

## `button-press-topdown`

<img src="images/button-press-topdown.png" alt="Press button top: the first state and the state at success" loading="lazy">

**Press button top.** Press a button from the top. Randomize button positions.

## `button-press-topdown-wall`

<img src="images/button-press-topdown-wall.png" alt="Press button top wall: the first state and the state at success" loading="lazy">

**Press button top wall.** Bypass a wall and press a button from the top. Randomize button positions.

## `button-press-wall`

<img src="images/button-press-wall.png" alt="Press button wall: the first state and the state at success" loading="lazy">

**Press button wall.** Bypass a wall and press a button. Randomize the button positions.

## `coffee-button`

<img src="images/coffee-button.png" alt="Get coffee: the first state and the state at success" loading="lazy">

**Get coffee.** Push a button on the coffee machine. Randomize the position of the coffee machine.

## `coffee-pull`

<img src="images/coffee-pull.png" alt="Pull mug: the first state and the state at success" loading="lazy">

**Pull mug.** Pull a mug from a coffee machine. Randomize the mug and the machine positions.

## `coffee-push`

<img src="images/coffee-push.png" alt="Push mug: the first state and the state at success" loading="lazy">

**Push mug.** Push a mug under a coffee machine. Randomize the mug and the machine positions.

## `dial-turn`

<img src="images/dial-turn.png" alt="Turn dial: the first state and the state at success" loading="lazy">

**Turn dial.** Rotate a dial 180 degrees. Randomize dial positions.

## `disassemble`

<img src="images/disassemble.png" alt="Disassemble nut: the first state and the state at success" loading="lazy">

**Disassemble nut.** Pick a nut out of the a peg. Randomize the nut positions.

## `door-close`

<img src="images/door-close.png" alt="Close door: the first state and the state at success" loading="lazy">

**Close door.** Close a door with a revolving joint. Randomize door positions.

## `door-lock`

<img src="images/door-lock.png" alt="Lock door: the first state and the state at success" loading="lazy">

**Lock door.** Lock the door by rotating the lock clockwise. Randomize door positions.

## `door-open`

<img src="images/door-open.png" alt="Open door: the first state and the state at success" loading="lazy">

**Open door.** Open a door with a revolving joint. Randomize door positions.

## `door-unlock`

<img src="images/door-unlock.png" alt="Unlock door: the first state and the state at success" loading="lazy">

**Unlock door.** Unlock the door by rotating the lock counter-clockwise. Randomize door positions.

## `drawer-close`

<img src="images/drawer-close.png" alt="Close drawer: the first state and the state at success" loading="lazy">

**Close drawer.** Push and close a drawer. Randomize the drawer positions.

## `drawer-open`

<img src="images/drawer-open.png" alt="Open drawer: the first state and the state at success" loading="lazy">

**Open drawer.** Open a drawer. Randomize drawer positions.

## `faucet-close`

<img src="images/faucet-close.png" alt="Turn off faucet: the first state and the state at success" loading="lazy">

**Turn off faucet.** Rotate the faucet clockwise. Randomize faucet positions.

## `faucet-open`

<img src="images/faucet-open.png" alt="Turn on faucet: the first state and the state at success" loading="lazy">

**Turn on faucet.** Rotate the faucet counter-clockwise. Randomize faucet positions.

## `hammer`

<img src="images/hammer.png" alt="Hammer: the first state and the state at success" loading="lazy">

**Hammer.** Hammer a screw on the wall. Randomize the hammer and the screw positions.

## `hand-insert`

<img src="images/hand-insert.png" alt="Insert hand: the first state and the state at success" loading="lazy">

**Insert hand.** Insert the gripper into a hole..

## `handle-press`

<img src="images/handle-press.png" alt="Press handle: the first state and the state at success" loading="lazy">

**Press handle.** Press a handle down. Randomize the handle positions.

## `handle-press-side`

<img src="images/handle-press-side.png" alt="Press handle side: the first state and the state at success" loading="lazy">

**Press handle side.** Press a handle down sideways. Randomize the handle positions.

## `handle-pull`

<img src="images/handle-pull.png" alt="Pull handle: the first state and the state at success" loading="lazy">

**Pull handle.** Pull a handle up. Randomize the handle positions.

## `handle-pull-side`

<img src="images/handle-pull-side.png" alt="Pull handle side: the first state and the state at success" loading="lazy">

**Pull handle side.** Pull a handle up sideways. Randomize the handle positions.

## `lever-pull`

<img src="images/lever-pull.png" alt="Pull lever: the first state and the state at success" loading="lazy">

**Pull lever.** Pull a lever down 90 degrees. Randomize lever positions.

## `peg-insert-side`

<img src="images/peg-insert-side.png" alt="Insert peg side: the first state and the state at success" loading="lazy">

**Insert peg side.** Insert a peg sideways. Randomize peg and goal positions.

## `peg-unplug-side`

<img src="images/peg-unplug-side.png" alt="Unplug peg: the first state and the state at success" loading="lazy">

**Unplug peg.** Unplug a peg sideways. Randomize peg positions.

## `pick-out-of-hole`

<img src="images/pick-out-of-hole.png" alt="Pick out of hole: the first state and the state at success" loading="lazy">

**Pick out of hole.** Pick up a puck from a hole. Randomize puck and goal positions.

## `pick-place`

<img src="images/pick-place.png" alt="Pick&place: the first state and the state at success" loading="lazy">

**Pick&place.** Pick and place a puck to a goal. Randomize puck and goal positions.

## `pick-place-wall`

<img src="images/pick-place-wall.png" alt="Pick&place w/ wall: the first state and the state at success" loading="lazy">

**Pick&place w/ wall.** Pick a puck, bypass a wall and place the puck. Randomize puck and goal positions.

## `plate-slide`

<img src="images/plate-slide.png" alt="Slide plate: the first state and the state at success" loading="lazy">

**Slide plate.** Slide a plate into a cabinet. Randomize the plate and cabinet positions.

## `plate-slide-back`

<img src="images/plate-slide-back.png" alt="Retrieve plate: the first state and the state at success" loading="lazy">

**Retrieve plate.** Get a plate from the cabinet. Randomize plate and cabinet positions.

## `plate-slide-back-side`

<img src="images/plate-slide-back-side.png" alt="Retrieve plate side: the first state and the state at success" loading="lazy">

**Retrieve plate side.** Get a plate from the cabinet sideways. Randomize plate and cabinet positions.

## `plate-slide-side`

<img src="images/plate-slide-side.png" alt="Slide plate side: the first state and the state at success" loading="lazy">

**Slide plate side.** Slide a plate into a cabinet sideways. Randomize the plate and cabinet positions.

## `push`

<img src="images/push.png" alt="Push: the first state and the state at success" loading="lazy">

**Push.** Push the puck to a goal. Randomize puck and goal positions.

## `push-back`

<img src="images/push-back.png" alt="Pull: the first state and the state at success" loading="lazy">

**Pull.** Pull a puck to a goal. Randomize puck and goal positions.

## `push-wall`

<img src="images/push-wall.png" alt="Push with wall: the first state and the state at success" loading="lazy">

**Push with wall.** Bypass a wall and push a puck to a goal. Randomize puck and goal positions.

## `reach`

<img src="images/reach.png" alt="Reach: the first state and the state at success" loading="lazy">

**Reach.** Reach a goal position. Randomize the goal positions.

## `reach-wall`

<img src="images/reach-wall.png" alt="Reach with wall: the first state and the state at success" loading="lazy">

**Reach with wall.** Bypass a wall and reach a goal. Randomize goal positions.

## `shelf-place`

<img src="images/shelf-place.png" alt="Place onto shelf: the first state and the state at success" loading="lazy">

**Place onto shelf.** Pick and place a puck onto a shelf. Randomize puck and shelf positions.

## `soccer`

<img src="images/soccer.png" alt="Soccer: the first state and the state at success" loading="lazy">

**Soccer.** Kick a soccer into the goal. Randomize the soccer and goal positions.

## `stick-pull`

<img src="images/stick-pull.png" alt="Pull with stick: the first state and the state at success" loading="lazy">

**Pull with stick.** Grasp a stick and pull a box with the stick. Randomize stick positions.

## `stick-push`

<img src="images/stick-push.png" alt="Push with stick: the first state and the state at success" loading="lazy">

**Push with stick.** Grasp a stick and push a box using the stick. Randomize stick positions..

## `sweep`

<img src="images/sweep.png" alt="Sweep: the first state and the state at success" loading="lazy">

**Sweep.** Sweep a puck off the table. Randomize puck positions.

## `sweep-into`

<img src="images/sweep-into.png" alt="Sweep into hole: the first state and the state at success" loading="lazy">

**Sweep into hole.** Sweep a puck into a hole. Randomize puck positions.

## `window-close`

<img src="images/window-close.png" alt="Close window: the first state and the state at success" loading="lazy">

**Close window.** Push and close a window. Randomize window positions.

## `window-open`

<img src="images/window-open.png" alt="Open window: the first state and the state at success" loading="lazy">

**Open window.** Push and open a window. Randomize window positions.
