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
    equal-contrib: true
    #maybe add orcid
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Dipartimento di Fisica Aldo Pontremoli, Università degli Studi di Milano, Milan, Italy
   index: 1
 - name: Independent Researcher, Italy
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
As mentioned before, SHTs are often the bottleneck of CMB data analysis pipelines, as the one implemented by Cosmoglobe[TOCITE] collaboration, based on the software Commander , which I contributed to with the work that led to the release of `HealpixMPI.jl`.

Given the significantly increasing amount of data produced by the most recent observational experiments, efficient algorithms alone are no longer enough to perform SHTs within acceptable run times and a parallel approach must be implemented.
Moreover, in the specific case of Cosmoglobe and Commander, the goal for the next years is to be able to run a full pipeline, and thus the SHTs performed in it, on a large HPC cluster *efficiently* employing at least $10^4$ cores.

In order to achieve this, an implementation of HEALPix allowing to perform spherical harmonics on a high number of cores, beyond the machine-size limitations, is unavoidably needed.

# The latest SHT engine: DUCC

As of the time of this paper being submitted, `Healpix.jl` relies on the SHTs provided by the C library `libsharp`[TOCITE]. However, since a couple of years ago, `libsharp`’s development has ceased and its functionalities have been included, as an SHT sub-module, in `DUCC` (Distinctively Useful Code Collection).

The timing between the development of `DUCC` and `HealpixMPI.jl` was quite lucky, as I became aware of the rising idea of a Julia interface for DUCC when I was about to start building a package out of my code.
This gave me the chance to swap the SHTs dependencies of `HealpixMPI.jl` from `libsharp`,
as initially planned, to `DUCC`; as well as helping Martin Reinecke, `DUCC`'s creator, with his new Julia interface.
This allowed `HealpixMPI.jl` to be already up-to-date with the state of the art of spherical harmonics upon it's first release.
In fact, for what concerns the SHTs, `DUCC`’s code is derived directly from `libsharp`, but has been significantly enhanced with the latest algorithmical improvements and the standard C++ multithreading implementation for *shared-memory* parallelization of the spherical harmonic transforms.

## Hybrid parallelization of the SHT

In order to run spherical harmonic transforsm on a large number `HealpixMPI.jl` was conceived to provides a hybrid parallelization of the computationally heaviest functionalities of `Healpix.jl`, through a simultaneous shared-memory (multithreading) and distributed-memory (MPI) parallel implementation.
For maximum efficiency, `HealpixMPI.jl` focuses on the

## Distributed data types


# Usage Example

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

I acknowledge crucial contributions from Maurizio Tomasi, main developer of `Healpix.jl` and supervisor of
my master thesis project at the University of Milan at the time of `HealpixMPI.jl`'s concept being born,
Martin Reinecke, main developer of DUCC[CITELINK], Hans Kristian Eriksen, the main developer of Commander,
Sigurd Naess and all the other members of Cosmoglobe collaboration.

# References
