# WaterSim
WaterSim is a physics engine aims to simulate water developed in taichi.

## Eulerian Methods
### Advection Algorithm
+ Semi-Lagrange Advection
+ Maccromack Advection (TODO)

### Projection Algorithm
+ CG Pressure Solver
+ IC(0) PCG Pressure Solver (Set MIC_blending to 0) (issue #12)
+ MIC(0) PCG Pressure Solver 
+ Geometric Multigrid Pressure Solver (TODO)

## Hybrid Eulerian-Lagrangian Methods (TODO)
+ PIC
+ Affine PIC
+ Poly PIC
+ FLIP
+ MPM