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
 - name: Independent Researcher, Country
   index: 2
date: 22 November 2023
bibliography: paper.bib

---

# Summary

The Julia package `HealpixMPI.jl` constitutes a natural extension of
`Healpix.jl` [TOCITE], providing an efficient parallelization of its sperical harmonic
transform (SHTs, for short) functionalities.
`Healpix.jl`, in turn, constitutes a Julia-only implementation of the HEALPix [TOCITE]
library, which provides one of the most used tasselation schemes of the two-sphere along with
a series of SHTs-related functions.
In brief, a spherical harmonic transform can be seen as a sort of two-dimensional Fourier transform defined on
the sphere, which can be used to decompose and analyze any spherically-symmetric field, reaching a wide variety of applications.

Moreover, the SHTs often constitute the computationally heaviest step of the scientific software they are part of.
For this reason, many efforts have been spent over the last decades to obtain the fastest and most efficient
possible SHTs implementations.
In such setting, parallel computing naturally comes into play, especially for heavy software to be run on high performance
computing (HPC) large clusters.
The main goal of the Julia package `HealpixMPI.jl`, presented in this paper, is to efficiently employ a high number of computing cores
in order to perform fast spherical harmonic transform.

**brief description of the main concept: -distribution of objects, - hybrid parallelization**


# Statement of need

Among a variety of applications, spherical harmonic transforms are particularly relevant for
the analysis of cosmic microwave background (CMB) radiation, which is one of the most active research field of recent cosmology.
CMB radiation is in fact very conveniently described as a temperature (and polarization) field on the sky sphere,
making spherical harmonics the most natural mathematical tool to analyze its measured signal.
Of course, from a computational point of view, CMB field measurements need to be discretized, requiring
a mathematically consistent pixelization of the sphere, and the functions defined on it.
This is exactly the goal HEALPix was aiming for, when more than two decades ago was released
quickly becoming the standard library for CMB numerical analysis.

Not surprisingly, the CMB is also the research field wherein `HealpixMPI.jl` was born.
As mentioned before, SHTs are often the bottleneck of CMB data analysis pipelines, e.g. the one
implemented by the Cosmoglobe[TOCITE] collaboration through the software Commander3[TOCITE].
Given the significantly increasing amount of data produced by the most recent observational experiments, efficient algorithms alone
are no longer enough to perform SHTs within acceptable run times and a parallel approach must be implemented.

`HealpixMPI.jl` tackles the problem by providing a hybrid parallelization of the
computationally heaviest functionalities of `Healpix.jl`, through a simultaneous
shared-memory (multithreading) and distributed-memory (MPI) parallel implementation.

# Main Features

## The latest SHT engine: DUCC

## Hybrid parallelization of the SHT

## Multi-platform support


# Mathematics

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

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

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
