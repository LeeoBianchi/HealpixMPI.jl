```@meta
DocTestSetup = quote
    using HealpixMPI
end
```

# Spherical Harmonic Transforms

The main target of HealpixMPI.jl, in terms of run time speed up, are the SHTs, which often represent the "bottle neck" of simulation codes or data analysis pipelines.

As mentioned in the introduction, HealpixMPI.jl relies on [ducc](https://gitlab.mpcdf.mpg.de/mtr/ducc)'s state-of-the-art algorithms for performing the spherical harmonic transforms.
In particular, its C++ functions are exploited for the computation of Legandre coefficients from alms and maps and vice versa, while the transposition of such coefficients between MPI tasks is entirely coded in Julia.

HealpixMPI.jl implements only the two exact spherical harmonic operators [`alm2map!`](@ref) (synthesis) and [`adjoint_alm2map!`](@ref) (adjoint synthesis), leaving to the user the task to implement the two corresponding **inverse** approximate operators, which can be done through many different approaches, from pixel and ring weights to conjugate gradient solver.
Both [`alm2map!`](@ref) and [`adjoint_alm2map!`](@ref) are implemented as overloads of the Healpix.jl's functions.

### From Alm to Map: synthesis operator

The synthesis SHT ([`alm2map`](@ref)) is used to compute a map from a set of $a_{\ell m}$ coefficients.
It is generally represented by the matrix operator $\mathrm{Y}$ which is defined through an exact summation as $$f(\theta, \phi) = \mathrm{Y} \, a_{\ell m} \quad \text{where} \quad f(\theta, \phi) = \sum_{\ell=0}^{\infty} \sum_{m=-\ell}^{\ell} a_{\ell m} Y_{\ell m} (\theta, \phi).$$

```@docs
alm2map!
```

### From Map to Alm: adjoint synthesis operator

The adjoint of the synthesis operator brings us from the map space to the harmonic space, as it is represented by the transpose $\mathrm{Y}^{\mathrm{T}}$.
Which is defined through: $$\mathrm{Y}^{\mathrm{T}} f(\theta, \phi) \equiv \sum_{i = 1}^{N_{\mathrm{pix}}} Y^*_{\ell m,\, i} \, f_i.$$

Note that this does not give directly the $a_{\ell m}$ coefficients, i.e.,  $$\mathrm{Y}^{\mathrm{T}} \mathrm{Y} \neq \mathbf{1}.$$ In fact, $$\mathrm{Y}^{-1} \simeq \mathrm{W}\, \mathrm{Y}^{\mathrm{T}}.$$Where $\mathrm{W}$ is a diagonal matrix whose non-zero elements are approximately constant and equal to $4 \pi / N_{\mathrm{pix}}$, depending on the map pixelization.
$\mathrm{Y}^{-1}$ is in fact an integral operator which must be approximated when implemented numerically.
```@docs
adjoint_alm2map!
```
