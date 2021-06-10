# Superlucent

This is my playground for practicing Vulkan programming, including:
- Physics simulation via Vulkan compute shader
- Rendering (hopefully ray tracing later) from compute shader results, without haevy CPU-GPU data transfers
- Graphical user interface, object-oriented

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
