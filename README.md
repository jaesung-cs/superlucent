# Superlucent

My Vulkan compute/graphics engine for learning purpose.

<p align="center">
  <img src="/results/particles_pbd.gif" width="400">
</p>

<p align="center">
  <img src="/results/cd_uniform_grid.gif" width="400">
  <img src="/results/cd_n2.gif" width="400">
</p>

(22^3 = 10,648 particles, collision detection with/without uniform grid)

It runs ~300 fps with 40^3 = 64k particles.

<p align="center">
  <img src="/results/wave.gif" width="400">
</p>

~220 fps with 64k particles.

## Goals

- Run physics simulation and rendering 100% on GPU, without heavy CPU-GPU transfers.

## Physics simulation

- Position Based Dynamics (PBD)

## Update notes

- See `/notes` directory for descriptions
- Until I lose interests...
