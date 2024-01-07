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

The spherical harmonic transforms, or SHTs for short, are a family of mathematical operators which often constitute the computationally heaviest step of scientific software in a variety of fields. For this reason, many efforts have been spent over the last decades to obtain the fastest and most efficient possible SHTs implementations.
In brief, a spherical harmonic transform can be seen as sort of two-dimensional Fourier transform defined on the sphere, which can be used for decomposing a spherically-symmetric field such as the temperature of the sky measured in a specific frequency range.
For this reason, SHTs are a crucial tool for the analysis of cosmic microwave background (CMB), which is one of the most active research field of recent cosmology.
As the resolution of the most recent observational experiments significantly increases, efficient algorithms alone are no longer enough to perform SHTs in acceptable run times and a parallel approach must be implemented.
Employing a high number of computing cores in the most efficient way in order to perform fast spherical harmonic transform operations is the main goal of the Julia package `HealpixMPI.jl`, presented in this paper.


# Statement of need

The Julia package `HealpixMPI.jl` constitutes a natural extension of the package
`Healpix.jl` [TOCITE], providing an efficient parallelization of its sperical harmonic
transform functionalities.
Where `Healpix.jl`, in turn, constitutes a Julia-only implementation of the HEALPix [TOCITE]
library, which provides one of the most used tasselation schemes of the two-sphere.
HEALPix is currently one of the most used libraries when performing numerical analysis of
the cosmic microwave background radiation.
In fact, the code existing under the hood of `HealpixMPI.jl` was born as a contribution to this
same very specific research field. However, I have decided to publish it in the form
of a new Julia package as it can be useful for a variety of projects other than mine.

As mentioned before, an efficient parallel implementation of the spherical harmonic
transforms is crucial in the CMB research field for obtaining feasible run times
when dealing with the most recent astrophyisical observational experiments which
provide increasingly high resolution.
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

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
