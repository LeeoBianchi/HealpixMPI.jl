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

The spherical harmonic transforms, or SHTs for short, are a family of mathematical operators 
which often constitute the computationally heaviest step of scientific software in a variety of fields.
For this reason, many efforts has been spent over the last decades to obtain the fastest and most efficient 
possible SHTs implementations.
In brief, a spherical harmonic transform can be seen as sort of two-dimensional Fourier transform defined on
the sphere, which can be used for decomposing a spherically-symmetric field such as the temperature
of the sky measured in a specific frequency range.
For this reason, SHTs are a crucial tool for the analysis of cosmic microwave background (CMB), which is one 
of the most active research field of recent cosmology. 
As the resolution of the most recent observational experiments significantly increases, efficient 
algorithms alone are no longer enough to perform SHTs in acceptable run times and a parallel approach must be implemented.
Employing a high number of computing cores in the most efficient way in order to perform fast spherical harmonic transform
operations is the main goal of the Julia package HealpixMPI.jl, presented in this paper.


# Statement of need

The Julia package HealpixMPI.jl
This package constitutes a natural extension of the package Healpix.jl, providing an MPI integration of its main functionalities, allowing for simultaneous shared-memory (multithreading) and distributed-memory (MPI) parallelization leading to high performance sperical harmonic transforms.

The Julia package HealpixMPI.jl wraps part of the code I wrote while carrying out my master thesis project at the Institute for
Theoretical Astrophysics of the University of Oslo. Even though my project was focused on
the very specific research field of the Cosmic Microwave Background (CMB), I have decided
to develop and publish this part of it in the form of a new Julia package, constituting an
extension of Healpix.jl, as it provides a simultaneous shared- and distributed-memory
parallelization of the main Spherical Harmonic Transforms (SHTs) functionalities which can
be useful for a variety of projects other than mine. In fact, one of the main goals of my work
was to allow my code to run efficiently on a high number of cores without multithreading’s
machine-size limitations. The SHTs are the computationally heaviest step of the algorithm
I implemented, where I had to concentrate the most of my efforts to obtain performance
improvements. In the following chapters I will go through the main features provided by
HealpixMPI.jl, as well as some brief usage examples and, eventually, the results of the
parallel benchmark tests I have carried out.

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

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
