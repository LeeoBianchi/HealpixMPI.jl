```@meta
DocTestSetup = quote
    using HealpixMPI
end
```

HealpixMPI.jl provides MPI-parallel versions, through overloads, of most of the other spherical-harmonic related functions of Healpix.jl.
Refer to Healpix.jl [documentation](https://ziotom78.github.io/Healpix.jl/stable/) for their description.

## Algebraic operations in harmonic space

HealpixMPI.jl provides overloads of the `Base` functions `+`, `-`, `*`, `/`,
as well as `LinearAlgebra.dot` (which embeds and `MPI.Allreduce` call), allowing to carry out these fundamental operations
element-wise in harmonic space directly.

```@docs
almxfl
almxfl!
Base.:+
Base.:-
Base.:*
Base.:/
LinearAlgebra.dot
```

## Power spectrum

Power spectrum components $C_{\ell}$ are encoded as Vector{T}.
HealpixMPI.jl implements overloads of Healpix.jl functions to compute a power spectrum from a set of `DAlm` ([`alm2cl`](@ref)) and to generate a set of `DAlm` from a power spectrum ([`synalm!`](@ref)).

```@docs
alm2cl
synalm!
```

## Distributing auxiliary arrays

It is often useful to make use of auxiliary arrays in pixel space, for which it is unnecessary to re-define a whole new map object, e.g., masks or noise covariance matrices.
HealpixMPI.jl provides an overload of `MPI.Scatter` to distribute the corresponding chunks of such arrays on the correct task.

```@docs
MPI.Scatter
```
