[![Codecov](https://codecov.io/gh/LeeoBianchi/HealpixMPI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LeeoBianchi/HealpixMPI.jl)
[![Build Status](https://github.com/LeeoBianchi/HealpixMPI.jl/workflows/Unit%20tests/badge.svg)](https://github.com/LeeoBianchi/HealpixMPI.jl/actions/workflows/UnitTest.yml)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://leeobianchi.github.io/HealpixMPI.jl/dev/)

<img src="docs/src/assets/logo.png" width="180">

# HealpixMPI.jl: an MPI-parallel implementation of the Healpix tessellation scheme in Julia

Welcome to HealpixMPI.jl, an MPI-parallel implementation of the main functionalities of [HEALPix](https://healpix.sourceforge.io/) spherical tessellation scheme, entirely coded in Julia.

This package constitutes a natural extension of the package [Healpix.jl](https://github.com/ziotom78/Healpix.jl), providing an MPI integration of its main functionalities, allowing for simultaneous shared-memory (multithreading) and distributed-memory (MPI) parallelization leading to high performance sperical harmonic transforms.

Read the full [documentation](https://leeobianchi.github.io/HealpixMPI.jl/dev) for further details.

## Installation

From the Julia REPL, run

````julia
import Pkg
Pkg.add("HealpixMPI")
````

## Usage Example

The example shows the necessary steps to set up and perform an MPI-parallel `alm2map` SHT with HealpixMPI.jl.

### Set up

We set up the necessary MPI communication and initialize Healpix.jl structures:
````julia
using MPI
using Random
using Healpix
using HealpixMPI

#MPI set-up
MPI.Init()
comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

#initialize Healpix structures
NSIDE = 64
lmax = 3*NSIDE - 1
if crank == root
  h_map = HealpixMap{Float64, RingOrder}(NSIDE)
  h_alm = Alm(lmax, lmax, randn(ComplexF64, numberOfAlms(lmax)))
else
  h_map = nothing
  h_alm = nothing
end
````

### Distribution

The distributed HealpixMPI.jl data types are filled through an overload of `MPI.Scatter!`:
````julia
#initialize empty HealpixMPI structures 
d_map = DMap{RR}(comm)
d_alm = DAlm{RR}(comm)

#fill them
MPI.Scatter!(h_map, d_map)
MPI.Scatter!(h_alm, d_alm)
````

### SHT

We perform the SHT through an overload of `Healpix.alm2map` and, if needed, we `Gather!` the result in a `HealpixMap`:

````julia
alm2map!(d_alm, d_map; nthreads = 16)
MPI.Gather!(d_map, h_map)
````

This allows the user to adjust at run time the number of threads to use, typically to be set to the number of cores of your machine.

## Run

In order to exploit MPI parallelization run the code through `mpirun` or `mpiexec` as
````shell
mpiexec -n {Ntask} julia {your_script.jl}
````

To run a code on multiple nodes, specify a machine file `machines.txt` as
````shell
mpiexec -machinefile machines.txt julia {your_script.jl}
````
