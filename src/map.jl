""" struct GeomInfoMPI{I <: Integer, T <: Real}

Information describing an MPI-distributed subset of a `HealpixMap`, contained in a `DistributedMap`.

An `GeomInfoMPI` type contains:
- `nside`: NSIDE parameter of the whole map.
- `maxnr`: maximum number of rings in the subsets, over the tasks involved.
- `comm`: MPI communicator used.
- `rings`: array of the ring indexes (w.r.t. the whole map) contained in the subset.
- `rstart`: array containing the index of the first pixel of each ring contained in the subset.
- `nphi`: array containing the number of pixels in every ring contained in the subset.
- `theta`: array of colatitudes (in radians) of the rings contained in the subset.
- `phi0`: array containing the values of the azimuth (in radians) of the first pixel in every ring.

"""
mutable struct GeomInfoMPI{T<:Real, I<:Integer}
    #communicator
    comm::MPI.Comm #FIXME: move in DistributedMap (?)

    #global info
    nside::I
    maxnr::I
    thetatot::Vector{T}

    #local info
    rings::Vector{I}
    rstart::Vector{I}
    nphi::Vector{I}
    theta::Vector{T}
    phi0::Vector{T}

    GeomInfoMPI{T,I}(nside::I, maxnr::I, thetatot::Vector{T}, rings::Vector{I}, rstart::Vector{I}, nphi::Vector{I}, theta::Vector{T}, phi0::Vector{T}, comm::MPI.Comm) where {T <: Real, I <: Integer} =
        new{T,I}(comm, nside, maxnr, thetatot, rings, rstart, nphi, theta, phi0)
end

GeomInfoMPI(nside::I, maxnr::I, rings::Vector{I}, rstart::Vector{I}, nphi::Vector{I}, theta::Vector{T}, phi0::Vector{T}, comm::MPI.Comm) where {T <: Real, I <: Integer} =
    GeomInfoMPI{T,I}(nside, maxnr, ring2theta(Resolution(nside)), rings, rstart, nphi, theta, phi0, comm)

#empty constructor
GeomInfoMPI{T,I}() where {T<:Real, I<:Integer} = GeomInfoMPI{T,I}(0, 0, Vector{T}(undef, 0), Vector{I}(undef, 0), Vector{I}(undef, 0), Vector{I}(undef, 0), Vector{T}(undef, 0), Vector{T}(undef, 0), MPI.COMM_NULL)
GeomInfoMPI() = GeomInfoMPI{Float64, Int64}()

"""
    Create an array of the colatitude in radians (theta) of each ring in `rings` for a map with resolution `res`.
    If no `rings` array is passed, the computation is performed on all the rings deducted from `res`.
"""
function ring2theta(rings::Vector{I}, res::Resolution) where {I<:Integer}
    theta = Vector{Float64}(undef, length(rings))
    ringinfo = RingInfo(0, 0, 0, 0, 0)
    @inbounds for i in 1:length(rings)
        getringinfo!(res, rings[i], ringinfo; full=true)
        theta[i] = ringinfo.colatitude_rad
    end
    theta
end

ring2theta(res::Resolution) = ring2theta(Vector{Int}(1:res.nsideTimesFour-1), res)

"""
    struct DistributedMap{T<:Number, I<:Integer}

A subset of a Healpix map, containing only certain rings (as specified in the `info` field).
The type `T` is used for the value of the pixels in a map, it must be a `Number` (usually float).
The type `AA` is used to store the array of pixels; typical types are `Vector`, `CUArray`, `SharedArray`, etc.

A `HealpixMap` type contains the following fields:

- `pixels::Vector{T}`: array of pixels composing the subset.
- `rings::Vector{I}`: array of the ring indexes included in the subset.
- `rstart::Vector{I}`: array of the indexes  (1-based) of the first pixel of each ring in `pixels`.
- `info::GeomInfo`: a `GeomInfo` object describing the HealpixMap subset.
- `comm::MPI.Comm`: communicator used to distribute the map.

The `GeomInfo` contained in `info` must match exactly the characteristic of the Map subset,
this can be constructed through the function `make_subset_healpix_geom_info`, for instance.

"""
mutable struct DistributedMap{T<:Real, I<:Integer}
    pixels::Vector{T}
    info::GeomInfoMPI{T,I}

    DistributedMap{T,I}(pixels::Vector{T}, info::GeomInfoMPI{T,I}) where {T<:Real, I<:Integer} =
        new{T,I}(pixels, info)
end

DistributedMap(pixels::Vector{T}, info::GeomInfoMPI{T,I}) where {T<:Real, I<: Integer} =
    DistributedMap{T,I}(pixels, info)

#empty constructors
DistributedMap{T,I}() where {T<:Real, I<:Integer} = DistributedMap{T,I}(Vector{T}(undef, 0), GeomInfoMPI{T,I}())
DistributedMap() = DistributedMap{Float64, Int64}()

function Healpix.numOfRings(nside::Integer)
    4*nside - 1
end

get_equator_idx(nside::Integer) = 2*nside
get_equator_idx(res::Resolution) = get_equator_idx(res.nside)

"""
    Return number of rings on specified task given total map resolution
    and communicator size according to Round Robin.
"""
function get_nrings_RR(eq_idx::Integer, task_rank::Integer, c_size::Integer)
    (task_rank < c_size) || throw(DomainError(0, "$task_rank can not exceed communicator size"))
    (eq_idx + c_size - 1 - task_rank) ÷ c_size * 2 - iszero(task_rank) #num of local rings we avoid counting the equator twice (assigned to 0-th task)
end

get_nrings_RR(res::Resolution, task_rank::Integer, c_size::Integer) = get_nrings_RR(get_equator_idx(res.nside), task_rank, c_size)

""" get_ring_pixels(map::HealpixMap{T,RingOrder,AA}, ring_info::RingInfo) where {T <: Real, AA <: AbstractArray{T,1}}
    get_ring_pixels(map::HealpixMap{T,RingOrder,AA}, ring_idx::Integer) where {T <: Real, AA <: AbstractArray{T,1}}

    Returns the pixels in `map` corresponding to the given `ring_info` or `ring_idx`.
"""
function get_ring_pixels(map::HealpixMap{T,RingOrder,AA}, ring_info::RingInfo) where {T <: Real, AA <: AbstractArray{T,1}}
    first_pix_idx = ring_info.firstPixIdx
    map[first_pix_idx:(first_pix_idx + ring_info.numOfPixels - 1)]
end

get_ring_pixels(map::HealpixMap{T,RingOrder,AA}, ring_idx::Integer) where {T <: Real, AA <: AbstractArray{T,1}} =
    get_ring_pixels(map, getringinfo(map.resolution, ring_idx; full=false))

"""
    Return array of rings on specified task (0-base index) given total map resolution
    and communicator size, ordered from the equator to the poles alternating N/S,
    according to Round Robin.
"""
function get_rindexes_RR(nside::Integer, t_rank::Integer, c_size::Integer)
    eq_idx = get_equator_idx(nside)
    nrings = get_nrings_RR(eq_idx, t_rank, c_s)
    rings = Vector{Int}(undef, nrings)
    @inbounds for i in 1:nrings
        k = (i - j) ÷ 2 #ring pair index (the same for each couple of corresponding north/south rings)
        ring = eq_idx - (-1)^(i + j) * (c_rank + k *c_size) #(-1)^i alternates rings north/south
        rings[i] = ring
    end
    rings
end

function get_rindexes_RR(local_nrings::Integer, eq_idx::Integer, t_rank::Integer, c_size::Integer)
    rings = Vector{Int}(undef, local_nrings)
    j = !iszero(t_rank)
    @inbounds for i in 1:local_nrings
        k = (i - j) ÷ 2 #ring pair index (the same for each couple of corresponding north/south rings)
        ring = eq_idx - (-1)^(i + j) * (t_rank + k *c_size) #(-1)^i alternates rings north/south
        rings[i] = ring
    end
    rings
end

#FIXME: create version which does not re-compute info in d_map
function ScatterMap_RR!(
    map::HealpixMap{T,RingOrder,Array{T,1}},
    d_map::DistributedMap{T}
    ) where {T <: Real, I <: Integer}

    stride = 1
    c_rank = MPI.Comm_rank(d_map.info.comm)
    c_size = MPI.Comm_size(d_map.info.comm)
    res = map.resolution
    nrings = get_nrings_RR(res, c_rank, c_size) #number of LOCAL rings
    #if we have too many MPI tasks, some will be empty
    (!iszero(nrings)) || throw(DomainError(0, "$c_rank-th MPI task has no rings."))
    rings = Vector{Int}(undef, nrings)
    rstart = Vector{Int}(undef, nrings)
    pixels = Vector{Vector{Float64}}(undef, nrings) #vector of rings
    theta = Vector{Float64}(undef, nrings) #colatitude of every ring
    phi0 = Vector{Float64}(undef, nrings)  #longitude of the first pixel of every ring #NOTE: how to do this with subset of maps?
    nphi = Vector{Float64}(undef, nrings)
    ringinfo = RingInfo(0, 0, 0, 0, 0) #initialize ring info object
    j = !iszero(c_rank) #index correction factor for when we have the equator (c_rank = 0, j=0), or not (c_rank > 0, j = 1)
    rst = 1 #keeps track of the indexes, to compute rstarts NOTE:(1-based)!!
    eq_idx = get_equator_idx(res)
    @inbounds for i in 1:nrings
        k = (i - j) ÷ 2 #ring pair index (the same for each couple of corresponding north/south rings)
        ring = eq_idx - (-1)^(i + j) * (c_rank + k *c_size) #(-1)^i+j alternates rings north/south
        rings[i] = ring
        getringinfo!(res, ring, ringinfo; full=true)
        theta[i] = ringinfo.colatitude_rad
        phi0[i] = pix2ang(map, ringinfo.firstPixIdx)[2]
        pixels[i] = get_ring_pixels(map, ringinfo)
        rstart[i] = rst
        nphi[i] = ringinfo.numOfPixels
        rst += ringinfo.numOfPixels
    end
    pixels = reduce(vcat, pixels) #FIXME:Is this the most efficient way?
    println("DistributedMap: I am task $c_rank of $c_size, I work on rings $rings of $(numOfRings(res)) \n")
    maxnr = (eq_idx%c_size == 0) ? get_nrings_RR(res, 0, c_size) : get_nrings_RR(res, 0, c_size)+1
    d_map.pixels = pixels
    d_map.info.comm = comm
    d_map.info.nside = res.nside
    d_map.info.maxnr = maxnr
    d_map.info.thetatot = ring2theta(res)
    d_map.info.rings = rings
    d_map.info.rstart = rstart
    d_map.info.nphi = nphi
    d_map.info.theta = theta
    d_map.info.phi0 = phi0
end

"""
    MPI.Scatter!(in_map::HealpixMap{T,RingOrder,Array{T,1}}, out_d_map::DistributedMap{T,I}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number, I <: Integer}
    MPI.Scatter!(in_alm::Nothing, out_d_map::DistributedMap{T,I}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number, I <: Integer}

    Distributes the `HealpixMap` object passed in input on the `root` task overwriting the
    `DistributedMap` objects passed on each task, according to the specified strategy
    (e.g. pass ":RR" for Round Robin).

    As in the standard MPI function, the `in_map` in input can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from the (potentially bulky) `Alm` object.

    # Arguments:
    - `in_map::HealpixMap{T,RingOrder,Array{T,1}}`: `HealpixMap` object to distribute over the MPI tasks.
    - `out_d_alm::DistributedMap{T,I}`: output `DistributedMap` object.

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input map after having performed the "scattering".
"""
function MPI.Scatter!(
    in_map::HealpixMap{T,RingOrder,Array{T,1}},
    out_d_map::DistributedMap{T};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Real, I <: Integer}

    comm = out_d_map.info.comm
    if MPI.Comm_rank(comm) == root
        MPI.bcast(in_map, root, comm)
    else
        in_map = MPI.bcast(nothing, root, comm)
    end

    if strategy == :RR #Round Robin, can add more.
        ScatterMap_RR!(in_map, out_d_map)
    end
    if clear
        in_map = nothing #free unnecessary copies of map
    end
end

function MPI.Scatter!(
    in_map::Nothing,
    out_d_map::DistributedMap{T,I};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Real, I <: Integer}

    comm = out_d_map.info.comm
    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input map on root task can NOT be `nothing`."))
    in_map = MPI.bcast(nothing, root, comm)

    if strategy == :RR #Round Robin, can add more.
        ScatterMap_RR!(in_map, out_d_map, comm)
    end
    if clear
        in_map = nothing #free unnecessary copies of map
    end
end

function MPI.Scatter!(
    in_map,
    out_d_map::DistributedMap{T,I},
    comm::MPI.Comm;
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Real, I <: Integer}
    out_d_map.info.comm = comm #overwrites comm in d_map
    MPI.Scatter!(in_map, out_d_map, strategy = strategy, root = root, clear = clear)
end

#######################################################################
#root task
function GatherMap_RR_root!(
    d_map::DistributedMap{T,I},
    map::HealpixMap{T,RingOrder,Array{T,1}},
    root::Integer
    ) where {T <: Real, I <: Integer}

    comm = d_map.info.comm
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    resolution = map.resolution
    ringinfo = RingInfo(0, 0, 0, 0, 0)
    rings = d_map.info.rings
    rstart = d_map.info.rstart
    @inbounds for ri in 1:d_map.info.maxnr
        if ri <= length(rings)
            getringinfo!(resolution, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[rstart[ri]:rstart[ri]+ringinfo.numOfPixels-1] #we select that ring's pixels
        else
            local_count = 0
            local_displ = 0
            pixels = Float64[]
        end
        counts = MPI.Gather(Int32(local_count), root, comm)
        displs = MPI.Gather(Int32(local_displ), root, comm)
        MPI.Barrier(comm) #FIXME: is it necessary?
        outbuf = MPI.VBuffer(map.pixels, counts, displs)
        MPI.Gatherv!(pixels, outbuf, root, comm)
    end
end

#for NON-ROOT tasks: no output
function GatherMap_RR_rest!(
    d_map::DistributedMap{T,I},
    root::Integer
    ) where {T <: Real, I <: Integer}

    comm = d_map.info.comm
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    resolution = Resolution(d_map.info.nside)
    ringinfo = RingInfo(0, 0, 0, 0, 0)
    rings = d_map.info.rings
    rstart = d_map.info.rstart
    @inbounds for ri in 1:d_map.info.maxnr
        if ri <= length(rings)
            getringinfo!(resolution, rings[ri], ringinfo; full=false)
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[rstart[ri]:rstart[ri]+ringinfo.numOfPixels-1] #we select that ring's pixels
        else
            local_count = 0
            local_displ = 0
            pixels = Float64[]
        end
        MPI.Gather(Int32(local_count), root, comm)
        MPI.Gather(Int32(local_displ), root, comm)
        MPI.Barrier(comm) #FIXME: is it necessary?
        MPI.Gatherv!(pixels, nothing, root, comm)
    end
end

"""
    MPI.Gather!(in_d_map::DistributedMap{T, I}, out_map::HealpixMap{T,RingOrder,Array{T,1}}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false)
    MPI.Gather!(in_d_map::DistributedMap{T, I}, out_map::Nothing, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false)

    Gathers the `DistributedMap` objects passed on each task overwriting the `HealpixMap`
    object passed in input on the `root` task according to the specified `strategy`
    (e.g. pass `:RR` for Round Robin). Note that the strategy must match the one used
    to "scatter" the map.

    As in the standard MPI function, the `out_map` can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from
    the (potentially bulky) `DistributedMap` object.

    # Arguments:
    - `in_d_map::DistributedMap{T, I}`: `DistributedMap` object to gather from the MPI tasks.
    - `out_map::HealpixMap{T,RingOrder,Array{T,1}}`: output `Map` object.

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `DistributedMap` after having performed the "scattering".
"""
function MPI.Gather!(
    in_d_map::DistributedMap{T,I},
    out_map::HealpixMap{T,RingOrder,Array{T,1}};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Real, I <: Integer}

    if strategy == :RR #Round Robin, can add more.
        if MPI.Comm_rank(in_d_map.info.comm) == root
            (out_map.resolution.nside == in_d_map.info.nside)||throw(DomainError(0, "nside not matching"))
            GatherMap_RR_root!(in_d_map, out_map, root)
        else
            GatherMap_RR_rest!(in_d_map, root)
        end
    end
    if clear
        in_d_map = nothing #free unnecessary copies of map
    end
end

#allows to pass nothing as output map on non-root tasks
function MPI.Gather!(
    in_d_map::DistributedMap{T,I},
    out_map::Nothing;
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Real, I <: Integer}

    (MPI.Comm_rank(in_d_map.info.comm) != root)||throw(DomainError(0, "output map on root task can not be `nothing`."))
    if strategy == :RR #Round Robin, can add more.
        GatherMap_RR_rest!(in_d_map, root)
    end
    if clear
        in_d_map = nothing #free unnecessary copies of map
    end
end

##########################################################

function AllgatherMap_RR!(
    d_map::DistributedMap{T,I},
    map::HealpixMap{T,RingOrder,Array{T,1}},
    root::Integer
    ) where {T <: Real, I <: Integer}

    comm = d_map.info.comm
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #each task can have at most root_nring + 1
    res = map.resolution
    ringinfo = RingInfo(0, 0, 0, 0, 0)
    rings = d_map.info.rings
    rstart = d_map.info.rstart
    @inbounds for ri in 1:d_map.info.maxnr
        if ri <= length(rings)
            getringinfo!(res, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[rstart[ri]:rstart[ri]+ringinfo.numOfPixels-1] #we select that ring's pixels
        else
            local_count = 0
            local_displ = 0
            pixels = Float64[]
        end
        counts = MPI.Allgather(Int32(local_count), comm)
        displs = MPI.Allgather(Int32(local_displ), comm)
        MPI.Barrier(comm) #FIXME: is it necessary?
        outbuf = MPI.VBuffer(map.pixels, counts, displs)
        MPI.Allgatherv!(pixels, outbuf, comm)
    end
end

"""
    MPI.Allgather!(in_d_map::DistributedMap{T,I}, out_map::HealpixMap{T,RingOrder,Array{T,1}}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number}

    Gathers the `DistributedMap` objects passed on each task overwriting the `out_map`
    object passed in input on EVERY task according to the specified `strategy`
    (e.g. pass `:RR` for Round Robin). Note that the strategy must match the one used
    to "scatter" the map.

    If the keyword `clear` is set to `true` it frees the memory of each task from
    the (potentially bulky) `DistributedMap` object.

    # Arguments:
    - `in_d_alm::DistributedMap{T,I}`: `DistributedMap` object to gather from the MPI tasks.
    - `out_d_alm::HealpixMap{T,RingOrder,Array{T,1}}`: output `HealpixMap` object to overwrite.

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Allgather!(
    in_d_map::DistributedMap{T,I},
    out_map::HealpixMap{T,RingOrder,Array{T,1}};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where  {T <: Real, I <: Integer}

    if strategy == :RR #Round Robin, can add more.
        AllgatherMap_RR!(in_d_map, out_map, root)
    end
    if clear
        in_d_map = nothing
    end
end
