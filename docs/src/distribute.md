```@meta
DocTestSetup = quote
    using HealpixMPI
end
```

# Distributed Classes

As mentioned in the introduction, HealpixMPI has the main purpose of providing an MPI parallelization of the main functionalities of [Healpix.jl](https://github.com/ziotom78/Healpix.jl), distributing maps and harmonic coefficients over the MPI tasks efficiently. 
This is made possible by the implementation of two data types: [`DistributedMap`](@ref) and [`DistributedAlm`](@ref), mirroring [`HealpixMap`](https://ziotom78.github.io/Healpix.jl/stable/mapfunc/#Healpix.HealpixMap) and [`Alm`](https://ziotom78.github.io/Healpix.jl/stable/alm/#Healpix.Alm) types of Healpix.jl respectively, and containing a well-defined subset of a map or harmonic coefficients, to be constructed on each MPI task.

```@docs
DistributedMap
DistributedAlm
```

An instance of `DistributedMap` (or `DistributedAlm`) embeds, whithin the field `info`, a [`GeomInfoMPI`](@ref) (or [`AlmInfoMPI`](@ref)) object. These latter, in turn, contain all the necessairy information about:

- The whole map geometry (or the whole set of harmonic coefficients).
- The composition of the *local* subset.
- the MPI communicator.

```@docs
GeomInfoMPI
AlmInfoMPI
```

## Initializing a Distributed type

The recommended way to construct a local subset of a map or harmonic coefficients, is to start with an instance of `HealpixMap` (in `RingOrder`) or `Alm` on the root task, and call one of the apposite overloads of the standard `MPI.Scatter!` function, provided by HealpixMPI.jl. 
Such function would in fact save the user the job of constructing all the required ancillary informations describing the data subset, doing so through efficient and tested methods.

While distributing a set of harmonic coefficients means that each MPI task will host a `DistributedAlm` object containing only the coefficients corresponding to some specific values of m, the distribution of a map is performed by rings.
Each MPI task will then host a `DistributedMap` object containing only the pixels composing some specified rings of the entire `HealpixMap`. 
Note that, for spherical harmonic transforms efficiency, it is recommended to assign pairs of rings with same latitude (i.e. symmetric w.r.t. the equator) to the same task, in order to preserve the geometric symmetry of the map.

It is also worth mentioning that one could find many different strategies to distribute a set of data over multiple MPI tasks.
So far, the only one implemented in HealpixMPI.jl, which should guarantee an adequate work balance between tasks, is the so-called "round robin" strategy: assuming $N$ MPI tasks, the map is distributed such that task $i$ hosts the map rings $i$, $i + N$, $i + 2N$, etc. (and their counterparts on the other hemisphere). 
Similarly, for the spherical harmonic coefficients, task $i$ would hold all coefficients for $m = i$, $i + N$, $i + 2 N$, etc.

```@docs
MPI.Scatter!
```

## Gathering data 

Analogously to `MPI.Scatter!`, HealpixMPI.jl also provides overloads of `MPI.Gather!` (and `MPI.Allgather!`).
These latter allow to re-group subsets of map or alm into a `HealpixMap` or `Alm` only on the root node (or on every MPI task involved).

```@docs
MPI.Gather!
MPI.Allgather!
```