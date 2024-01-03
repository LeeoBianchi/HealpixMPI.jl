
"""
    struct GeomInfoMPI

Information describing an MPI-distributed subset of a `HealpixMap`, contained in a `DMap`.

A `GeomInfoMPI` type contains:
- `comm`: MPI communicator used.
- `nside`: NSIDE parameter of the whole map.
- `maxnr`: maximum number of rings in the subsets, over the tasks involved.
- `thetatot`: array of the colatitudes of the whole map ordered by task first and RR within each task
- `rings`: array of the ring indexes (w.r.t. the whole map) contained in the subset.
- `rstart`: array containing the 1-based index of the first pixel of each ring contained in the subset.
- `nphi`: array containing the number of pixels in every ring contained in the subset.
- `theta`: array of colatitudes (in radians) of the rings contained in the subset.
- `phi0`: array containing the values of the azimuth (in radians) of the first pixel in every ring.

"""
mutable struct GeomInfoMPI
    #communicator
    comm::MPI.Comm

    #global info
    nside::Int
    maxnr::Int
    thetatot::Vector{Float64}

    #local info
    rings::Vector{Int}
    rstart::Vector{Int}
    nphi::Vector{Int}
    theta::Vector{Float64}
    phi0::Vector{Float64}

    GeomInfoMPI(nside::I, maxnr::I, thetatot::Vector{T}, rings::Vector{I}, rstart::Vector{I}, nphi::Vector{I}, theta::Vector{T}, phi0::Vector{T}, comm::MPI.Comm) where {T <: Real, I <: Integer} =
        new(comm, nside, maxnr, thetatot, rings, rstart, nphi, theta, phi0)
end
#without thetatot
GeomInfoMPI(nside::I, maxnr::I, rings::Vector{I}, rstart::Vector{I}, nphi::Vector{I}, theta::Vector{T}, phi0::Vector{T}, comm::MPI.Comm) where {T <: Real, I <: Integer} =
    GeomInfoMPI(nside, maxnr, ring2theta(Healpix.Resolution(nside)), rings, rstart, nphi, theta, phi0, comm)
#only comm
GeomInfoMPI(comm::MPI.Comm) =
    GeomInfoMPI(0, 0, Vector{Float64}(undef, 0), Vector{Int}(undef, 0), Vector{Int}(undef, 0), Vector{Int}(undef, 0), Vector{Float64}(undef, 0), Vector{Float64}(undef, 0), comm)
#empty constructor
GeomInfoMPI() =
    GeomInfoMPI(0, 0, Vector{Float64}(undef, 0), Vector{Int}(undef, 0), Vector{Int}(undef, 0), Vector{Int}(undef, 0), Vector{Float64}(undef, 0), Vector{Float64}(undef, 0), MPI.COMM_NULL)

"""
    abstract type AbstractDMap

Abstract type to allow multiple dispatch.
"""
abstract type AbstractDMap end

"""
    struct DMap{S<:Strategy, T<:Real, N} <: AbstractDMap

A subset of a Healpix map, containing only certain rings (as specified in the `info` field).
The type `T` is used for the value of the pixels in a map, it must be a `Real` (usually float).
The signature field `N` represents the number of components in the `DMap` object, it can only be `1`
for a spin-0 field and `2` otherwise.

A `DMap` type contains the following fields:

- `pixels::Matrix{T}`: array of pixels composing the subset, dimensions are `(npixels, ncomp)`.
- `info::GeomInfo`: a `GeomInfo` object describing the HealpixMap subset.

The `GeomInfoMPI` contained in `info` must match exactly the characteristic of the Map subset,
this is already automatically constructed when [`MPI.Scatter!`](@ref) is called, reason why this
method for initializing a `DMap` is reccomended.

"""
mutable struct DMap{S<:Strategy, T<:Real} <: AbstractDMap
    pixels::Matrix{T} #alias of Array{T,2}
    info::GeomInfoMPI

    DMap{S,T}(pixels::Matrix{T}, info::GeomInfoMPI) where {S<:Strategy, T<:Real} = new{S,T}(pixels, info)
end

#comm-only constructors
DMap{S,T}(comm::MPI.Comm) where {T<:Real, S<:Strategy} = DMap{S,T}(Matrix{T}(undef, 0, 1), GeomInfoMPI(comm))
DMap{S}(comm::MPI.Comm) where {S<:Strategy} = DMap{S,Float64}(comm)

#lazy constructors
DMap{S,T}() where {T<:Real, S<:Strategy} = DMap{S,T}(Matrix{T}(undef, 0, 1), GeomInfoMPI())
DMap{S}() where {S<:Strategy} = DMap{S,Float64}()

#Overload of size operator
Base.size(m::DMap{S,T}) where {S,T} = size(m.pixels)
Base.parent(m::DMap{S,T}) where {S<:Strategy,T} = m.pixels

"""
    get_nrings_RR(eq_idx::Integer, task_rank::Integer, c_size::Integer)
    get_nrings_RR(res::Resolution, task_rank::Integer, c_size::Integer)

Return number of rings on specified task given total map resolution
and communicator size according to Round Robin.
"""
function get_nrings_RR(eq_idx::Integer, task_rank::Integer, c_size::Integer)::Integer
    (task_rank < c_size) || throw(DomainError(0, "$task_rank can not exceed communicator size"))
    (eq_idx + c_size - 1 - task_rank) ÷ c_size * 2 - iszero(task_rank) #num of local rings we avoid counting the equator twice (assigned to 0-th task)
end
get_nrings_RR(res::Healpix.Resolution, task_rank::Integer, c_size::Integer)::Integer = get_nrings_RR(Healpix.getEquatorIdx(res.nside), task_rank, c_size)

"""
    get_rindexes_RR(local_nrings::Integer, eq_idx::Integer, t_rank::Integer, c_size::Integer)
    get_rindexes_RR(nside::Integer, t_rank::Integer, c_size::Integer)

Return array of rings on specified task (0-base index) given total map resolution
and communicator size, ordered from the equator to the poles alternating N/S,
according to Round Robin.
"""
function get_rindexes_RR(local_nrings::Integer, eq_idx::Integer, t_rank::Integer, c_size::Integer)::Vector{Int}
    rings = Vector{Int}(undef, local_nrings)
    j = !iszero(t_rank)
    @inbounds for i in 1:local_nrings
        k = (i - j) ÷ 2 #ring pair index (the same for each couple of corresponding north/south rings)
        rings[i] = eq_idx - (-1)^(i + j) * (t_rank + k *c_size) #(-1)^i alternates rings north/south
    end
    rings
end

function get_rindexes_RR(nside::Integer, t_rank::Integer, c_size::Integer)::Vector{Int}
    eq_idx = Healpix.getEquatorIdx(nside)
    nrings = get_nrings_RR(eq_idx, t_rank, c_size)
    get_rindexes_RR(nrings, eq_idx, t_rank, c_size)
end

"""
    get_rindexes_tot_RR(eq_index::Integer, c_size::Integer)

Return array of ring indexes ordered by task first and RR within each task.
"""
function get_rindexes_tot_RR(eq_index::Integer, c_size::Integer)
    filled = 1 #we keep track of how much of rindexes we already have filled
    rindexes = Vector{Int}(undef, eq_index*2-1)
    for t_rank in 0:c_size-1
        nring = get_nrings_RR(eq_index, t_rank, c_size)
        rindexes[filled:filled+nring-1] = get_rindexes_RR(nring, eq_index, t_rank, c_size)
        filled += nring
    end
    rindexes
end

"""
Internal function implementing a "Round Robin" strategy (note the type rquirement on d_map).

Here the input map is supposed to be on every task as a copy.
The input map object is broadcasted by `MPI.Scatter!`.
"""
function ScatterMap!(
    map::Healpix.HealpixMap{T,Healpix.RingOrder},
    d_map::DMap{RR,T} #Round Robin, can add more as overloads
    ) where {T <: Real}

    #stride = 1
    c_rank = MPI.Comm_rank(d_map.info.comm)
    c_size = MPI.Comm_size(d_map.info.comm)
    res = map.resolution
    nrings = get_nrings_RR(res, c_rank, c_size) #number of LOCAL rings
    #if we have too many MPI tasks, some will be empty
    (!iszero(nrings)) || throw(DomainError(0, "$c_rank-th MPI task has no rings."))
    rings = Vector{Int}(undef, nrings)  #vector of local rings
    rstart = Vector{Int}(undef, nrings)
    theta = Vector{Float64}(undef, nrings) #colatitude of every ring
    phi0 = Vector{Float64}(undef, nrings)  #longitude of the first pixel of every ring
    nphi = Vector{Float64}(undef, nrings)
    ringinfo = Healpix.RingInfo(0, 0, 0, 0, 0) #initialize ring info object
    j = !iszero(c_rank) #index correction factor for when we have the equator (c_rank = 0, j=0), or not (c_rank > 0, j = 1)
    rst = 1 #keeps track of the indexes, to compute rstarts
    eq_idx = Healpix.getEquatorIdx(res)
    d_map.pixels = Matrix{T}(undef, 0, 1) #initialize pixels
    @inbounds for i in 1:nrings
        k = (i - j) ÷ 2 #ring pair index (the same for each couple of corresponding north/south rings)
        ring = eq_idx - (-1)^(i + j) * (c_rank + k *c_size) #(-1)^i+j alternates rings north/south
        rings[i] = ring
        Healpix.getringinfo!(res, ring, ringinfo; full=true)
        theta[i] = ringinfo.colatitude_rad
        phi0[i] = Healpix.pix2ang(map, ringinfo.firstPixIdx)[2]
        d_map.pixels = cat(d_map.pixels, Healpix.getRingPixels(map, ringinfo), dims=1) #append pixels
        rstart[i] = rst
        nphi[i] = ringinfo.numOfPixels
        rst += ringinfo.numOfPixels
    end
    println("DMap: I am task $c_rank of $c_size, I work on $(length(rings)) rings of $(Healpix.numOfRings(res))")
    maxnr = get_nrings_RR(res, 0, c_size)+1
    d_map.info.nside = res.nside
    d_map.info.maxnr = maxnr
    thetatot = Healpix.ring2theta(res)
    rindexes = get_rindexes_tot_RR(eq_idx, c_size)
    d_map.info.thetatot = thetatot[rindexes] #colatitudes ordered by task first and RR within each task
    d_map.info.rings = rings
    d_map.info.rstart = rstart
    d_map.info.nphi = nphi
    d_map.info.theta = theta
    d_map.info.phi0 = phi0
end

function ScatterMap!(
    polmap::Healpix.PolarizedHealpixMap{T,Healpix.RingOrder},
    d_map::DMap{RR,T}, #Round Robin, can add more as overloads
    ) where {T <: Real}

    ScatterMap!(polmap.q, d_map) #first call the scalar function to build the info in d_map along with q map
    d_map.pixels = cat(d_map.pixels, Array{T}(undef, size(d_map)), dims = 2) #extra columns for u map
    done = 0
    @inbounds for ring in d_map.info.rings
        new_pixels = Healpix.getRingPixels(polmap.u, ring)
        @views d_map.pixels[done+1:done+length(new_pixels), 2] = new_pixels #fill u component
        done += length(new_pixels)
    end
end

import MPI: Scatter!, Gather!, Allgather!

"""
    Scatter!(in_map::Union{Healpix.HealpixMap{T1, Healpix.RingOrder}, Healpix.PolarizedHealpixMap{T1, Healpix.RingOrder}}, out_d_map::DMap{S,T2,I}; root::Integer = 0, clear::Bool = false) where {T1<:Real, T2<:Real, S<:Strategy}
    Scatter!(in_map::Healpix.PolarizedHealpixMap{T1, Healpix.RingOrder}, out_d_map::DMap{S,T2}, out_d_pol_map::DMap{S,T2}; root::Integer = 0, clear::Bool = false) where {T1<:Real, T2<:Real, S<:Strategy}
    Scatter!(in_map::Nothing, out_d_map::DMap{S,T}; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}
    Scatter!(in_map::Nothing, out_d_map::DMap{S,T}, out_d_pol_map::DMap{S,T}; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}
    Scatter!(in_map, out_d_map::DMap{S,T}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}
    Scatter!(in_map, out_d_map::DMap{S,T}, out_d_pol_map::DMap{S,T}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}

Distributes the `HealpixMap` object passed in input on the `root` task overwriting the
`DMap` objects passed on each task, according to the specified strategy.

There are two options to distribute a `HealpixMap` which represents a polarized field:
- To pass an input `PolarizedHealpixMap` object and only one output `DMap` object, in this case the latter will be exclusively filled with its `q` and `u` components.
- To pass an input `PolarizedHealpixMap` object and two output `DMap` object, in this case one will contain the `i` map, while the second `q` and `u`.
This is so that the resulting `DMap` objects can be directly passed to the sht functions which only accept in input the intensity component for a scalar transform and two polarization components for a spinned transform.

As in the standard MPI function, the `in_map` in input can be `nothing` on non-root tasks,
since it will be ignored anyway.

If the keyword `clear` is set to `true` it frees the memory of each task from the (potentially bulky) `HealpixMap` object.

# Arguments:
- `in_map: `HealpixMap` or `PolarizedHealpixMap` object to distribute over the MPI tasks.
- `out_d_map::DMap{S,T}`: output `DMap` object.

# Keywords:
- `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
- `clear::Bool`: if true deletes the input map after having performed the "scattering".
""" #(by default ":RR" for Round Robin).
function Scatter!(
    in_map::Union{Healpix.HealpixMap{T1, Healpix.RingOrder}, Healpix.PolarizedHealpixMap{T1, Healpix.RingOrder}},
    out_d_pol_map::DMap{S,T2};
    root::Integer = 0,
    clear::Bool = false
    ) where {T1<:Real, T2<:Real, S<:Strategy}

    comm = out_d_pol_map.info.comm
    if MPI.Comm_rank(comm) == root
        MPI.bcast(in_map, root, comm)
    else
        in_map = MPI.bcast(nothing, root, comm)
    end

    ScatterMap!(in_map, out_d_pol_map)

    if clear
        in_map = nothing #free unnecessary copies of map
    end
end

function Scatter!(
    in_map::Healpix.PolarizedHealpixMap{T1, Healpix.RingOrder},
    out_d_map::DMap{S,T2},
    out_d_pol_map::DMap{S,T2};
    root::Integer = 0,
    clear::Bool = false
    ) where {T1<:Real, T2<:Real, S<:Strategy}

    Scatter!(in_map.i, out_d_map, root = root, clear = clear)
    Scatter!(in_map, out_d_pol_map, root = root, clear = clear)
end

function Scatter!(
    in_map::Nothing,
    out_d_map::DMap{S,T};
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Real, S<:Strategy}

    comm = out_d_map.info.comm
    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input map on root task can NOT be `nothing`."))
    in_map = MPI.bcast(nothing, root, comm)

    ScatterMap!(in_map, out_d_map)

    if clear
        in_map = nothing #free unnecessary copies of map
    end
end

function Scatter!(
    in_map::Nothing,
    out_d_map::DMap{S,T},
    out_d_pol_map::DMap{S,T};
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Real, S<:Strategy}

    Scatter!(in_map, out_d_map, root = root, clear = clear)
    Scatter!(in_map, out_d_pol_map, root = root, clear = clear)
end

function Scatter!(
    in_map,
    out_d_map::DMap{S,T},
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Real, S<:Strategy}
    out_d_map.info.comm = comm #overwrites comm in d_map
    MPI.Scatter!(in_map, out_d_map, root = root, clear = clear)
end

function Scatter!(
    in_map,
    out_d_map::DMap{S,T},
    out_d_pol_map::DMap{S,T},
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Real, S<:Strategy}
    out_d_map.info.comm = comm #overwrites comm in d_map
    out_d_pol_map.info.comm = comm
    MPI.Scatter!(in_map, out_d_map, out_d_pol_map, root = root, clear = clear)
end

#######################################################################
"""
Internal function implementing a "Round Robin" strategy.

"""
function GatherMap!(
    d_map::DMap{RR,T},
    map::Healpix.HealpixMap{T,Healpix.RingOrder},
    root::Integer,
    comp::Integer
    ) where {T <: Real}

    (map.resolution.nside == d_map.info.nside)||throw(DomainError(0, "nside not matching"))
    (size(d_map.pixels, 2) >= comp) || throw(DomainError(4, "not enough components in DMap"))

    comm = d_map.info.comm
    resolution = Healpix.Resolution(d_map.info.nside)
    ringinfo = Healpix.RingInfo(0, 0, 0, 0, 0)
    rings = d_map.info.rings
    rstart = d_map.info.rstart
    @inbounds for ri in 1:d_map.info.maxnr
        if ri <= length(rings)
            Healpix.getringinfo!(resolution, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[rstart[ri]:rstart[ri]+ringinfo.numOfPixels-1, comp] #we select that ring's pixels
        else
            local_count = 0
            local_displ = 0
            pixels = Float64[]
        end
        counts = MPI.Gather(Int32(local_count), root, comm)
        displs = MPI.Gather(Int32(local_displ), root, comm)
        outbuf = MPI.VBuffer(map.pixels, counts, displs)
        MPI.Gatherv!(pixels, outbuf, root, comm)
    end
end

#non-root
function GatherMap!(
    d_map::DMap{RR,T},
    map::Nothing,
    root::Integer,
    comp::Integer
    ) where {T <: Real}

    (size(d_map.pixels, 2) >= comp) || throw(DomainError(4, "not enough components in DMap"))
    (MPI.Comm_rank(d_map.info.comm) != root)||throw(DomainError(0, "Output map on root task can not be nothing."))

    comm = d_map.info.comm
    resolution = Healpix.Resolution(d_map.info.nside)
    ringinfo = Healpix.RingInfo(0, 0, 0, 0, 0)
    rings = d_map.info.rings
    rstart = d_map.info.rstart
    @inbounds for ri in 1:d_map.info.maxnr
        if ri <= length(rings)
            Healpix.getringinfo!(resolution, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[rstart[ri]:rstart[ri]+ringinfo.numOfPixels-1, comp] #we select that ring's pixels
        else
            local_count = 0
            local_displ = 0
            pixels = Float64[]
        end
        MPI.Gather(Int32(local_count), root, comm)
        MPI.Gather(Int32(local_displ), root, comm)
        MPI.Gatherv!(pixels, nothing, root, comm)
    end
end

"""
    Gather!(in_d_map::DMap{S,T}, out_map::Union{Healpix.HealpixMap{T,Healpix.RingOrder}, Nothing}, comp::Integer; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}
    Gather!(in_d_map::DMap{S,T}, out_map; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}
    Gather!(in_d_map::DMap{S,T}, in_d_pol_map::DMap{S,T}, out_map::Healpix.PolarizedHealpixMap{T,Healpix.RingOrder}; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}
    Gather!(in_d_map::DMap{S,T}, in_d_pol_map::DMap{S,T}, out_map::Nothing; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy}

Gathers the `DMap` objects passed on each task overwriting the `HealpixMap` or `PolarizedHealpixMap`
object passed in input on the `root` task according to the specified `Strategy`.

Similarly to the standard MPI function, the `out_map` can be `nothing` on non-root tasks,
since it will be ignored anyway.

If the keyword `clear` is set to `true` it frees the memory of each task from
the (potentially bulky) `DMap` object.

# Arguments:
- `in_d_map::DMap{S,T}`: `DMap` object to gather from the MPI tasks.
- `out_map`: output `HealpixMap` or `PolarizedHealpixMap` object.

# Optional:
- `comp::Integer`: Specify which component (column) from the pixel matrix in `DMap` is to be gathered, defaulted to 1.

# Keywords:
- `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
- `clear::Bool`: if true deletes the input `DMap` after having performed the "scattering".
"""
function Gather!(
    in_d_map::DMap{S,T},
    out_map::Union{Healpix.HealpixMap{T,Healpix.RingOrder}, Nothing},
    comp::Integer;
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Real, S<:Strategy}

    GatherMap!(in_d_map, out_map, root, comp)

    if clear
        in_d_map = nothing #free unnecessary copies of map
    end
end
#defaults comp to 1
Gather!(in_d_map::DMap{S,T}, out_map; root::Integer = 0, clear::Bool = false) where {T<:Real, S<:Strategy} =
    Gather!(in_d_map, out_map, 1, root = root, clear = clear)

#for polarized maps
function Gather!(
    in_d_map::DMap{S,T},
    in_d_pol_map::DMap{S,T},
    out_map::Healpix.PolarizedHealpixMap{T,Healpix.RingOrder};
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Real, S<:Strategy}
    Gather!(in_d_map, out_map.i, 1, root = root, clear = clear)      #I
    Gather!(in_d_pol_map, out_map.q, 1, root = root, clear = clear)  #Q
    Gather!(in_d_pol_map, out_map.u, 2, root = root, clear = clear)  #U
end

#non-root tasks where it can be nothing
function Gather!(
    in_d_map::DMap{S,T},
    in_d_pol_map::DMap{S,T},
    out_map::Nothing;
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Real, S<:Strategy}
    Gather!(in_d_map, nothing, 1, root = root, clear = clear)      #I
    Gather!(in_d_pol_map, nothing, 1, root = root, clear = clear)  #Q
    Gather!(in_d_pol_map, nothing, 2, root = root, clear = clear)  #U
end

##########################################################
"""
    Internal function implementing a "Round Robin" strategy.
"""
function AllgatherMap!(
    d_map::DMap{RR,T},
    map::Healpix.HealpixMap{T,Healpix.RingOrder},
    comp::Integer
    ) where {T<:Real}

    (size(d_map.pixels, 2) >= comp) || throw(DomainError(4, "not enough components in DMap"))
    comm = d_map.info.comm
    #each task can have at most root_nring + 1
    res = map.resolution
    ringinfo = Healpix.RingInfo(0, 0, 0, 0, 0)
    rings = d_map.info.rings
    rstart = d_map.info.rstart
    @inbounds for ri in 1:d_map.info.maxnr
        if ri <= length(rings)
            Healpix.getringinfo!(res, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[rstart[ri]:rstart[ri]+ringinfo.numOfPixels-1, comp] #we select that ring's pixels
        else
            local_count = 0
            local_displ = 0
            pixels = Float64[]
        end
        counts = MPI.Allgather(Int32(local_count), comm)
        displs = MPI.Allgather(Int32(local_displ), comm)
        outbuf = MPI.VBuffer(map.pixels, counts, displs)
        MPI.Allgatherv!(pixels, outbuf, comm)
    end
end

"""
    Allgather!(in_d_map::DMap{S,T}, out_map::Healpix.HealpixMap{T,Healpix.RingOrder} comp::Integer; clear::Bool = false) where {T<:Real, S<:Strategy}
    Allgather!(in_d_map::DMap{S,T}, out_map::Healpix.HealpixMap{T,Healpix.RingOrder}; clear::Bool = false) where {T<:Real, S<:Strategy}
    Allgather!(in_d_map::DMap{S,T}, out_map::Healpix.PolarizedHealpixMap{T,Healpix.RingOrder}; clear::Bool = false) where {T<:Real, S<:Strategy}

Gathers the `DMap` objects passed on each task overwriting the `out_map`
object passed in input on EVERY task according to the specified `strategy`
(by default `:RR` for Round Robin). Note that the strategy must match the one used
to "scatter" the map.

If the keyword `clear` is set to `true` it frees the memory of each task from
the (potentially bulky) `DMap` object.

# Arguments:
- `in_d_map::DMap{S,T}`: `DMap` object to gather from the MPI tasks.
- `out_d_map`: output `HealpixMap` or `PolarizedHealpixMap` object to overwrite.

# Optional:
- `comp::Integer`: Specify which component (column) from the pixel matrix in `DMap` is to be gathered, defaulted to 1.

# Keywords:
- `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function Allgather!(
    in_d_map::DMap{S,T},
    out_map::Healpix.HealpixMap{T,Healpix.RingOrder},
    comp::Integer;
    clear::Bool = false
    ) where  {T<:Real, S<:Strategy}

    (out_map.resolution.nside == in_d_map.info.nside)||throw(DomainError(0, "nside not matching"))
    AllgatherMap!(in_d_map, out_map, comp)

    if clear
        in_d_map = nothing
    end
end

#defaults comp to 1
Allgather!(in_d_map::DMap{S,T}, out_map::Healpix.HealpixMap{T,Healpix.RingOrder}; clear::Bool = false) where  {T<:Real, S<:Strategy} =
    Allgather!(in_d_map, out_map, 1, clear = clear)

function Allgather!(
    in_d_map::DMap{S,T},
    in_d_pol_map::DMap{S,T},
    out_map::Healpix.PolarizedHealpixMap{T,Healpix.RingOrder};
    clear::Bool = false
    ) where  {T<:Real, S<:Strategy}
    Allgather!(in_d_map, out_map.i, 1, clear = clear)       #I
    Allgather!(in_d_pol_map, out_map.q, 1, clear = clear)   #Q
    Allgather!(in_d_pol_map, out_map.u, 2, clear = clear)   #U
end

"""
    ≃(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Real}

Similarity operator, returns `true` if the two arguments have matching `info` objects.
"""
function ≃(map₁::DMap{S,T}, map₂::DMap{S,T}) where {S<:Strategy, T<:Real}
    (&)((map₁.info.comm == map₂.info.comm),
        (map₁.info.nside == map₂.info.nside),
        (map₁.info.maxnr == map₂.info.maxnr),
        (map₁.info.thetatot == map₂.info.thetatot),
        (map₁.info.rings == map₂.info.rings),
        (map₁.info.rstart == map₂.info.rstart),
        (map₁.info.nphi == map₂.info.nphi),
        (map₁.info.theta == map₂.info.theta),
        (map₁.info.phi0 == map₂.info.phi0))
end

## DMap Algebra
import Base: +, -, *, /

+(a::DMap{S,T}, b::DMap{S,T}) where {T<:Real, S<:Strategy} =
    DMap{S,T}(a.pixels .+ b.pixels, a ≃ b ? a.info : throw(DomainError(0,"info not matching")))
-(a::DMap{S,T}, b::DMap{S,T}) where {T<:Real, S<:Strategy} =
    DMap{S,T}(a.pixels .- b.pixels, a ≃ b ? a.info : throw(DomainError(0,"info not matching")))
*(a::DMap{S,T}, b::DMap{S,T}) where {T<:Real, S<:Strategy} =
    DMap{S,T}(a.pixels .* b.pixels, a ≃ b ? a.info : throw(DomainError(0,"info not matching")))
/(a::DMap{S,T}, b::DMap{S,T}) where {T<:Real, S<:Strategy} =
    DMap{S,T}(a.pixels ./ b.pixels, a ≃ b ? a.info : throw(DomainError(0,"info not matching")))

+(a::DMap{S,T}, b::Number) where {T<:Real, S<:Strategy} = DMap{S,T}(a.pixels .+ b, a.info)
-(a::DMap{S,T}, b::Number) where {T<:Real, S<:Strategy} = a + (-b)
*(a::DMap{S,T}, b::Number) where {T<:Real, S<:Strategy} = DMap{S,T}(a.pixels .* b, a.info)
/(a::DMap{S,T}, b::Number) where {T<:Real, S<:Strategy} = DMap{S,T}(a.pixels ./ b, a.info)

+(a::Number, b::DMap{S,T}) where {T<:Real, S<:Strategy} = b + a
*(a::Number, b::DMap{S,T}) where {T<:Real, S<:Strategy} = b * a
/(a::Number, b::DMap{S,T}) where {T<:Real, S<:Strategy} = DMap{S,T}(a ./ b.pixels, b.info)
