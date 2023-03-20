[![codecov](https://img.shields.io/codecov/c/github/LeeoBianchi/HealpixMPI.jl?style=plastic)](https://app.codecov.io/gh/LeeoBianchi/HealpixMPI.jl)
![Tests](https://github.com/LeeoBianchi/HealpixMPI.jl/actions/workflows/UnitTests.yml/badge.svg)

<img src="docs/src/assets/logo.png" width="180">


# HealpixMPI.jl: an MPI-parallel implementation of the Healpix tessellation scheme in Julia

Welcome to HealpixMPI.jl, an MPI-parallel implementation of the main functionalities of [HEALPix](https://healpix.sourceforge.io/) spherical tessellation scheme, entirely coded in Julia.

This package constitutes a natural extension of the package [Healpix.jl](https://github.com/ziotom78/Healpix.jl), providing an MPI integration of its main functionalities, allowing for simultaneous shared-memory (multithreading) and distributed-memory (MPI) parallelization leading to high performance sperical harmonic transforms.

Read the full [documentation](https://leeobianchi.github.io/HealpixMPI.jl/dev).
