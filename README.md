# gHCM
[![Build Status](https://travis-ci.org/OlivierHamel/gHCM.svg?branch=master)](https://travis-ci.org/OlivierHamel/gHCM)
GPU Eikonal Solver

gHCM is a semi-GPU eikonal solver similar to (p)HCM and FIM. It was conceived as part of a term project for a course on parallel algorithms. Briefly, it is effectively FIM but instead of solving all blocks in the open set every iteration, only the first _n_ lowest-minimum-time blocks are solved. (_n_ is a tunable parameter.) This solves the case where FIM might overwhelm the GPU with blocks which could never (at that time) contribute to the final solution. In exchange, we must maintain some sort of priority structure on the CPU but this is practically always a net-win for non-trivial eikonals and large data sets.

The project uses OpenCL 1.2, due to locally-available hardware constraints.