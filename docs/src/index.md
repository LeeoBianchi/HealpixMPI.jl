```@meta
DocTestSetup = quote
    using HealpixMPI
end
```

# HealpixMPI.jl: an MPI-parallel implementation of the Healpix tessellation scheme in Julia

Welcome to [HealpixMPI.jl](https://github.com/LeeoBianchi/HealpixMPI.jl), an MPI-parallel implementation of [Healpix.jl](https://github.com/ziotom78/Healpix.jl), an Healpix spherical tessellation scheme written entirely in Julia.

This package constitutes a natural extension of the package Healpix.jl, providing an MPI integration of the main functionalities, allowing for high-performances and better scaling on high resolutions.

More specifically, three main features can be highlighted:
- **High-performance spherical harmonic transforms**. This library relies on [ducc](https://gitlab.mpcdf.mpg.de/mtr/ducc)'s state-of-the-art algorithms for performing fast and efficient SHTs.
- **Best scaling features**. The simultaneous usage of modern C++ multithreading (provided by ducc), for shared-memory parallelization) and MPI (for distributed-memory parallelization) allows, in principle, for the best scaling possible on SHTs, i.e. 1-to-1 correspondence between m values (or map rings) and processes, if the code is run on a sufficiently large cluster of machines.
Note that this would be in practice impossible to achieve for high resolutions without the usage of MPI, which allows to distribute our maps and harmonic coefficients over different computing nodes, since generally such a high number of computing cores is never available on the same machine.
- **Cross-platform support**. This package maintains the same multi-platform compatibility of Healpix.jl, thanks to the cross-platform support of [MPI.jl](https://github.com/JuliaParallel/MPI.jl), Ducc0.jl (ducc's wrapper package providing Julia interface) and Julia itself.

## Documentation

The documentation was built using [Documenter.jl](https://github.com/JuliaDocs).

```@example
using Dates # hide
println("Documentation built on $(now()) using Julia $(VERSION).") # hide
```

## Index

```@index
```
