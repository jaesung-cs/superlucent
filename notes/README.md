# Superlucent

This is my playground for practicing Vulkan programming, including:
- Physics simulation via Vulkan compute shader
- Rendering (hopefully ray tracing later) from compute shader results, without haevy CPU-GPU data transfers
- Graphical user interface, object-oriented

## Fluid siulation (July 18, 2021)

In commit `54f6439`,

<p align="center">
  <img src="/results/fluid_simulation.jpg">
</p>
(80 fps with 32k particles, x1/10 timestep for stable simulation)

<p align="center">
  <img src="/results/fluid1.gif" width="400">
  <img src="/results/fluid2.gif" width="400">
</p>
(220 fps with 16k particles)

Implemented fluid simulation, translating [InteractiveComputerGraphics/PositionBasedDynamics](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics) repo.
- Neighbor search with max 60 neighbors per particle (lower than that causes incorrect density estimation)
- Support radius h = 4r
- Gaussian-like kernel function
- No boundary particles

Thing to fix:
- Use boundary particles for correct density estimation
  - Akinci et al., "Versatile Rigid-Fluid Coupling for Incompressible SPH"
- Improve simulation performance and stability

## Code refactoring (July 11, 2021)

In commit `e2ca714`,

As code refactoring, GPU particle simulator is moved and renamed as a separate project `vkpbd`.
The purpose is to move the responsibility of managing memories and buffers from class to the application and the user.

There are still more codes for refactoring:
- member variable naming convention: from `snake_case` to `camelCase` for consistency with Vulkan.
- delete CPU simulator and other dead codes
- change `vkpbd` from header-only to static library

And fixed a fatal bug in `uniform_grid.glsl`
- `CellIndex()` was returning an integer offset grid coordinates, causing `Bound()` function calculate wrong values.

Next topic is probably to continue fluid simulation, as well as code refactoring.

## Code refactoring (July 3, 2021)

In commit `8abf0e8`,

Tested CPU-to-GPU transfer with transfer semaphore every frame, not so fast.
FPS drops from 500 to 250, i.e. ~2ms transfer time.
Possible solution is using a transfer thread, a transfer queue and triple buffering, which requires optimization in data transfer, and is not a goal in this project.

Deleted fluid simulation code which was not working.

## Solver update with linked list of collision pairs (June 12, 2021)

In PR #2 (commit `4815997`),

The bottleneck was `solve-delta-x`: was running in O(nm), where m the number of collisions.
Particles with same radius can have up to 12 (= k) contacts per particle.
With linked lists the time complexity reduces from O(nm) = O(n^2 k) to O(nk).
Now it runs ~300 fps with 40^3 = 64k particles.

## Uniform grid (June 12, 2021)

<p align="center">
  <img src="/results/cd_uniform_grid.gif" width="400">
  <img src="/results/cd_n2.gif" width="400">
</p>

In PR #1 (commit `dc1db23`), I made a uniform grid for particle collision detection.
- Used a hash function, allowing hash collisions.
- Used a linked list with atomic add/exchange operations.
- Why ivec3 -> uvec3 cast in hash function not working as intended? I had to add some positive integers instead of casting.

Sadly, the bottleneck wasn't collision detection.
- About 10% speed up in the whole simulation steps.
- Reducing the number of PBD solver iterations hugely increased the performance.
- The current maximum number of particles on the boundary of stable simulation is 22^3 = 10,648, running ~120 fps.

## Basic implementation (June 10, 2021)

[![basic](http://img.youtube.com/vi/g6oJ62bBLPc/0.jpg)](http://www.youtube.com/watch?v=g6oJ62bBLPc) \
(Link to Youtube, 4096 particles, ~30 fps with many contact constraints)

By commit `9281091` I present the basic pipeline of my Vulkan physics simulation and rendering, with the most basic components:
- Physics simulation
  - I made many `particle_*.comp` compute shaders, composing an (Extended) Particle-Based Dynamics ((X)PBD) physics simulation.
  - Collision_detection: exhaustive O(n^2) particle collision detection, probably the performance bottleneck. I plan to upgrade as the next step with uniform grid.
  - Solve lambda and x: follows PBD equations.
  - Velocity update: no velocity solve yet, so need to change to simulate elastic collision responses.
  - `vkCmdDispatchIndirect`: between compute stage and draw indirect stage (why?)
- Rendering
  - Instance rendering is used, with sphere vertex buffer and particle position index buffer.
- Physics simulation and rendering command submissions are not optimized yet; they are submitted sequentially, frame by frame.
