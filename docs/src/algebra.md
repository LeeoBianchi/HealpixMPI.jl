```@meta
DocTestSetup = quote
    using HealpixMPI
end
```

# Spherical Harmonic Transforms

The main target of HealpixMPI.jl, in terms of run time speed up, are the SHTs, which often represent the "bottle neck" of simulation codes or data analysis pipelines.

As mentioned in the introduction, HealpixMPI.jl relies on [ducc](https://gitlab.mpcdf.mpg.de/mtr/ducc)'s state-of-the-art algorithms for performing the spherical harmonic transforms.
In particular, its C++ functions are exploited for the computation of Legandre coefficients from alms and maps and vice versa, while the transposition of such coefficients between MPI tasks is entirely coded in Julia.

The two SHTs are implemented as overloads of Healpix.jl's functions.

```@docs
Healpix.alm2map!
Healpix.adjoint_alm2map!
```

## Other Spherical Harmonic functionalities

HealpixMPI.jl provides parallel versions, through overloads, of most of the other spherical-harmonic related functions of Healpix.jl.
Refer to Healpix.jl [documentation](https://ziotom78.github.io/Healpix.jl/stable/) for their description.

```@docs
Healpix.alm2cl
Healpix.synalm!
Healpix.almxfl!
```