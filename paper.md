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
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Dipartimento di Fisica Aldo Pontremoli, Università degli Studi di Milano, Milan, Italy
   index: 1
   
 - name: Institute of Theoretical Astrophysics, University of Oslo, Blindern, Oslo, Norway
   index: 2
date: 22 November 2023
bibliography: paper.bib

---

# Summary

The Julia package `HealpixMPI.jl` constitutes a natural extension of `Healpix.jl`[@Healpix_jl], providing an efficient parallelization of its sperical harmonic transform (SHTs, for short) functionalities.
`Healpix.jl`, in turn, constitutes a Julia-only implementation of the HEALPix[@HEALPix] library, which provides one of the most used tasselation schemes of the two-sphere along with a series of SHTs-related functions.
In brief, a spherical harmonic transform can be seen as a sort of two-dimensional Fourier transform defined on the sphere, which can be used to decompose and analyze any spherically-symmetric field, becoming an essential tool for solving a wide variety of problems.

However, the SHTs are in general computationally expensive operations and thus they often constitute the *bottleneck* of the scientific software they are part of.
For this reason, many efforts have been spent over the last couple of decades to obtain the fastest and most efficient possible SHTs implementations.
In such setting, parallel computing naturally comes into play, especially for heavy software to be run on high performance computing (HPC) large clusters.
The main goal of the Julia package `HealpixMPI.jl`, presented in this paper, is to efficiently employ a high number of computing cores in order to perform fast spherical harmonic transforms.
The principal features implemented to achieve this, together with a statement of need and a brief usage example are presented in this paper.

![Healpix Logo \label{fig:logo}](docs/src/assets/logo.png){width=40%}

# Statement of need

Among a variety of applications, spherical harmonic transforms are particularly relevant for the analysis of cosmic microwave background (CMB) radiation, which is one of the most active research field of recent cosmology.
CMB radiation is in fact very conveniently described as a temperature (and polarization) field on the sky sphere, making spherical harmonics the most natural mathematical tool to analyze its measured signal.
On the other hand, from a computational point of view, CMB field measurements need of course to be discretized, requiring a mathematically consistent pixelization of the sphere, and the functions defined on it.
This is exactly the goal HEALPix was targeting, when more than two decades ago was released, quickly becoming the standard library for CMB numerical analysis.

Not surprisingly, the cosmic microwave background is also the research context wherein `HealpixMPI.jl` concept was born.
As mentioned before, SHTs are often the bottleneck of CMB data analysis pipelines, as the one implemented by Cosmoglobe [@Watts_2023] collaboration, based on the software Commander [@Eriksen_2004], which I contributed to with the work that led to the release of `HealpixMPI.jl`.

Given the significantly increasing amount of data produced by the most recent observational experiments, efficient algorithms alone are no longer enough to perform SHTs within acceptable run times and a parallel approach must be implemented.
Moreover, in the specific case of Cosmoglobe and Commander, the goal for the next years is to be able to run a full pipeline, and thus the SHTs performed in it, on a large HPC cluster *efficiently* employing at least $10^4$ cores.

In order to achieve this, an implementation of HEALPix allowing to perform spherical harmonics on a high number of cores, beyond the machine-size limitations, is unavoidably needed.

# The latest SHT engine: DUCC

As of the time of this paper being submitted, `Healpix.jl` relies on the SHTs provided by the C library `libsharp`[@libsharp]. However, since a few years ago, `libsharp`’s development has ceased and its functionalities have been included, as an SHT sub-module, in `DUCC` [@ducc], acronym of "Distinctively Useful Code Collection".

The timing between the development of `HealpixMPI.jl` and a Julia interface for `DUCC` has been quite lucky.
This allowed `HealpixMPI.jl` to be already up-to-date with the state of the art of spherical harmonics upon it's first release.
In fact, for what concerns the SHTs, `DUCC`’s code is derived directly from `libsharp`, but has been significantly enhanced with the latest algorithmical improvements as well as the standard C++ multithreading implementation for *shared-memory* parallelization of the spherical harmonic transforms.

# Hybrid parallelization of the SHT

In order to run spherical harmonic transforms on a large number of cores, `HealpixMPI.jl` provides a hybrid parallel design, based on a simultaneous usage of multithreading and MPI, for shared- and distributed-memory parallelization respectively.

In particular, multithreading is provided by `DUCC` for its spherical harmonic transforms 

# Usage Example




# Acknowledgements

I acknowledge crucial contributions from Maurizio Tomasi, my supervisor at the University of Milan at the time of `HealpixMPI.jl`'s concept being born, Martin Reinecke, Hans Kristian Eriksen and Sigurd Næss; as well as the support I received from all the other members of Cosmoglobe collaboration.

# References
