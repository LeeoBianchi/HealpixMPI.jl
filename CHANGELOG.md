# HEAD

# HealpixMPI v1.0.0

**Merged pull requests:**
- Add spin-2 sht (#4) (@LeeoBianchi)

**What's new:**
- Polarization support for SHT
- Polarization support for distributed datatypes DAlm & DMap
- Polarization support for MPI.Scatter!, MPI.Gather! and MPI.Allgather! overload methods, to allow direct portability to and from Healpix.jl
- DAlm & DMap have now abstract super-types

**Braking changes:**
- Removed the Integer type specification in the signature of AlmInfoMPI and GeomInfoMPI, and of DAlm and DMap consequently. 

# Version 0.1.0

**Merged pull requests:**
- Make an abstract type for strategy specification (#1) (@LeeoBianchi)
- add auxiliary leg arguments in sht & docstrings (#2) (@LeeoBianchi)

# Version 0.1.0-beta

-   First public release
