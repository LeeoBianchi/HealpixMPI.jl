---
title: 'HealpixMPI.jl: an MPI-parallel implementation of the Healpix tessellation scheme in Julia'
tags:
  - Julia
  - SHT
  - Healpix
  - parallel computing
  - cosmology
authors:
  - name: Leo A. Bianchi
    orcid: 0009-0002-6351-5426
    affiliation: "1, 2" # (Multiple affiliations must be quoted)

affiliations:
 - name: Dipartimento di Fisica Aldo Pontremoli, Università degli Studi di Milano, Milan, Italy            
   index: 1

 - name: Institute of Theoretical Astrophysics, University of Oslo, Blindern, Oslo, Norway
   index: 2
date: 10 January 2024
bibliography: paper.bib

---

# Summary

Spherical Harmonic Transforms (SHTs) can be seen as the spherical, two-dimensional, counterpart of Fourier Transforms, casting real-space data to the spectral domain and vice versa.
As in Fourier analysis a function is decomposed into a set of amplitude coefficients, through an SHT, any spherically-symmetric field defined in real space can be decomposed into a set of complex harmonic coefficients $a_{\ell, m}$, commonly referred to as alms, each quantifying the contribution of the corresponding spherical harmonic function.

SHTs are important for a wide variety of theoretical and practical scientific application, including particle physics, astrophysics and cosmology.
However, the SHTs are in general computationally expensive operations and thus often constitute the *bottleneck* of the scientific software they are part of.
For this reason, many efforts have been spent over the last couple of decades to obtain fast and efficient SHTs implementations.
In such a setting, parallel computing naturally comes into play, especially for heavy software to be run on large High Performance Computing (HPC) clusters.

The Julia package `HealpixMPI.jl` constitutes an extension package of `Healpix.jl` [@Healpix_jl], providing an efficient parallelization of its SHTs functionalities.
`Healpix.jl` is a Julia-only implementation of the HEALPix [@HEALPix] library, which provides one of the most used tasselation schemes of the two-sphere along with a series of SHTs-related functions.

The main goal of the Julia package `HealpixMPI.jl`, presented in this paper, is to efficiently employ a high number of computing cores in order to perform fast spherical harmonic transforms.
The key features implemented to achieve this, together with a statement of need the results of a parallel scaling test are presented in this paper.

![`HealpixMPI.jl`'s logo \label{fig:logo}](figures/logo.png){width=30%}

# Statement of need

Among a variety of applications, spherical harmonic transforms are particularly relevant for the analysis of cosmic microwave background (CMB) radiation, which is one of the most active research field of recent cosmology.
CMB radiation is in fact very conveniently described as a temperature (and polarization) field on the sky sphere, making spherical harmonics the most natural mathematical tool to analyze its measured signal.
On the other hand, from a computational point of view, CMB field measurements need of course to be discretized, requiring a mathematically consistent pixelization of the sphere, and the functions defined on it.
This is exactly the goal HEALPix was targeting, when more than two decades ago was released, quickly becoming the standard library for CMB numerical analysis.
HEALPix code can be of course used for a wider variety of applications, but its bond with CMB analysis has always been particularly strong, especially given the research interests of its main authors.

Not surprisingly, the cosmic microwave background is also the research context wherein `HealpixMPI.jl` was born.
SHTs are often the computational bottleneck of CMB data analysis pipelines, as the one implemented by Cosmoglobe [@Watts_2023] collaboration, based on the software Commander [@Eriksen_2004], which I have been contributing to with the work that led to the release of `HealpixMPI.jl`.

Given the significantly increasing amount of data produced by the most recent observational experiments, efficient algorithms alone are no longer enough to perform SHTs within acceptable run times and a parallel approach must be implemented.
In the specific case of Cosmoglobe and Commander, the goal for the next years is to be able to run a full pipeline, and thus the SHTs performed in it, on a large HPC cluster *efficiently* employing at least $10^4$ cores.

In order to achieve this, an implementation of HEALPix allowing to perform spherical harmonics on a high number of cores, beyond the machine-size limitations, is unavoidably needed.

# The latest SHT engine: DUCC

As of the time of this paper being submitted, `Healpix.jl` relies on the SHTs provided by the C library `libsharp` [@libsharp]. However, since a few years ago, `libsharp`’s development has ceased and its functionalities have been included, as an SHT sub-module, in `DUCC` [@ducc], acronym of "Distinctively Useful Code Collection".

The timing between the development of `HealpixMPI.jl` and a Julia interface for `DUCC` has been quite lucky.
This allowed `HealpixMPI.jl` to be already up-to-date with the state of the art of spherical harmonics upon its first release.
In fact, for what concerns the SHTs, `DUCC`’s code is derived directly from `libsharp`, but has been significantly enhanced with the latest algorithmical improvements as well as the standard C++ multithreading implementation for *shared-memory* parallelization of the spherical harmonic transforms.

# Hybrid parallelization of the SHT

To run SHTs on a large number of cores, i.e. on a HPC cluster, `HealpixMPI.jl` provides a hybrid parallel design, based on a simultaneous usage of multithreading and MPI, for shared- and distributed-memory parallelization respectively, as shown in figure \autoref{fig:hybrid}.

![Multi-node computing cluster representation. The optimal way to parallelize operations such as the SHTs on a cluster of computers is to employ MPI to share the computation *between* the available nodes, assigning one MPI task per node, and multithreading to parallelize *within* each node, involving as many CPUs as locally available. Figure taken from www.comsol.com. \label{fig:hybrid}](figures/hybrid_parallel.png){width=70%}

In the case of ‘HealpixMPI.jl’, native C++ multithreading is provided by `DUCC` for its spherical harmonic transforms by default; while the MPI interface is entirely coded in Julia and based on the package `MPI.jl` [@MPI].

Moreover, the MPI parallelization requires data to be distributed across the MPI tasks.
As shown in the usage examples, this is implemented by mirroring `Healpix.jl`'s classes with two new *distributed* data types: `DAlm` and `DMap`, encoding the harmonic coefficients and a pixelized representation of the spherical field respectively.

# Usage example

An usage example with all the necessary steps to set up and perform an MPI-parallel `alm2map` SHT can be found in the front page of `HealpixMPI.jl`'s [repository](https://github.com/LeeoBianchi/HealpixMPI.jl).

In addition, refer to [Jommander](https://github.com/LeeoBianchi/Jommander.jl), a parallel and Julia-only CMB Gibbs Sampler, for an example of code based on `HealpixMPI.jl`.


# Scaling results

This section shows the results of some parallel benchmark tests conducted on `HealpixMPI.jl`.
In particular, a strong-scaling scenario is analyzed: given a problem of fixed size, the wall time improvement is measured as the number of cores exploited in the computation is increased.

To obtain a reliable measurement of massively parallel spherical harmonics wall time is certainly nontrivial: especially for tests implying a high number of cores, intermittent operating system activity can significantly distort the measurement of short time scales.
For this reason, the benchmark tests were carried out by timing a batch of 20 `alm2map` + `adjoint_alm2map` SHT pairs.
For reference, the scaling shown here is relative to unpolarized spherical harmonics with $\mathrm{N}_\mathrm{side} = 4096$ and $\ell_{\mathrm{max}} = 12287$ and were carried out on the [Hyades cluster](https://www.mn.uio.no/astro/english/services/it/help/basic-services/compute-resources.html) of the University of Oslo.
The benchmark results are quantified as the wall time multiplied by the total number of cores, shown in a 3d-plot (figure \autoref{fig:bench}) as a function of the number of local threads and MPI tasks (one per node).

![The measured wall time is multiplied by the total number of cores used, and plotted as a function of the number of local threads and MPI tasks (one per node) used. The total number of cores corresponding to each column is of course given by the product of these two quantities. \label{fig:bench}](figures/3DBench.png){width=75%}

Increasing the number of threads on a single core, for which no MPI communication is needed, leads to an almost-ideal scaling up to $\sim 50$ cores. For 60 and higher local threads we start observing a slight slowdown, probably given by the many threads simultaneously trying to access the same memory, hitting its bandwidth limit.
While switching to a multi-node setup, we introduce, as expected, an overhead given by the necessary MPI communication whose size, unfortunately, remains constant as we increase the number of local threads, leading to the ramp-shape, along the "local threads"-axis, shown by the plot.
However, the overhead size do scale down, even if not perfectly, when we increase the number of nodes, as the size of the locally stored data will linearly decrease.
This can be seen by the fact that, along the "nodes"-axis, $t_{\mathrm{wall}} \times N_{\mathrm{cores}}$ remains approximately constant.

# Acknowledgements

The development of `HealpixMPI.jl`, as a part of my master thesis, has been funded by the University of Milan, through a "Thesis Abroad Grant".
Moreover, I acknowledge significant contributions to my project from Maurizio Tomasi, Martin Reinecke, Hans Kristian Eriksen and Sigurd Næss; as well as the support I received from all the members of Cosmoglobe collaboration during my stay at the Institute of Theoretical Astrophysics of the University of Oslo.

# References
