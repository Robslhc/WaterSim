# WaterSim
WaterSim is a physics engine aims to simulate water developed in taichi.

## Eulerian Methods
### Advection Algorithm
+ Semi-Lagrange Advection

### Projection Algorithm
+ CG Pressure Solver
+ IC(0) PCG Pressure Solver (Set MIC_blending to 0) (issue #12)
+ MIC(0) PCG Pressure Solver 
+ Geometric Multigrid Pressure Solver 

## Hybrid Eulerian-Lagrangian Methods (TODO)
+ PIC
+ FLIP
+ Affine PIC