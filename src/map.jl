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
mutable struct DistributedMap{T<:Number, I<:Integer}
    pixels::Vector{T}
    rings::Vector{I}   #######
    rstarts::Vector{I} #######FIXME: maybe embed these two in GeomInfo
    info::GeomInfo
    comm::MPI.Comm

    DistributedMap{T,I}(pixels::Vector{T}, rings::Vector{I}, rstarts::Vector{I}, info::GeomInfo, comm::MPI.Comm) where {T<:Number, I<:Integer} =
        new{T,I}(pixels, rings, rstarts, info, comm)
end

DistributedMap(pixels::Vector{T}, rings::Vector{I}, rstarts::Vector{I}, info::GeomInfo, comm::MPI.Comm) where {T<:Number, I<:Integer} =
    DistributedMap{T,I}(pixels, rings, rstarts, info, comm)

#constructs also the info object, assuming stride=1
DistributedMap(pixels::Vector{T}, rings::Vector{I}, rstarts::Vector{I}, nside::Integer, comm::MPI.Comm) where {T<:Number, I<:Integer} =
    DistributedMap{T,I}(pixels, rings, rstarts, make_subset_healpix_geom_info(nside, 1, rings), comm)

#empty constructors
DistributedMap{T,I}() where {T<:Number, I<:Integer} = DistributedMap{T,I}(Vector{T}(undef, 0), Vector{I}(undef, 0), Vector{I}(undef, 0), GeomInfo(Ptr{Cvoid}()), MPI.COMM_WORLD)
DistributedMap() = DistributedMap{Float64, Int}()

function ScatterMap_RR!(
    map::HealpixMap{T,RingOrder,Array{T,1}},
    d_map::DistributedMap{T,I},
    comm::MPI.Comm
    ) where {T <: Number, I <: Integer}

    stride = 1
    c_rank = MPI.Comm_rank(comm)
    c_size = MPI.Comm_size(comm)
    res = map.resolution
    eq_idx = get_equator_idx(res)
    nrings = (eq_idx + c_size - 1 - c_rank) ÷ c_size * 2 - iszero(c_rank) #num of local rings we avoid counting the equator twice (assigned to 0-th task)
    #if we have too many MPI tasks, some will be empty
    (!iszero(nrings)) || throw(DomainError(0, "$c_rank-th MPI task has no rings."))
    rings = Vector{Int}(undef, nrings)
    rstarts = Vector{Int}(undef, nrings)
    pixels = Vector{Vector{ComplexF64}}(undef, nrings) #vector of rings
    ringinfo = RingInfo(0, 0, 0, 0, 0) #initialize ring info object
    j = !iszero(c_rank) #index correction factor for when we have the equator (c_rank = 0, j=0), or not (c_rank > 0, j = 1)
    rstart = 1 #keeps track of the indexes, to compute rstarts (1-based)
    @inbounds for i in 1:nrings
        k = (i - j)÷ 2 #ring pair index (the same for each cuple of corresponding north/south rings)
        ring = eq_idx - (-1)^(i + j) * (c_rank + k *c_size) #(-1)^i alternates rings north/south
        rings[i] = ring
        getringinfo!(res, ring, ringinfo; full=false)
        pixels[i] = get_ring_pixels(map, ringinfo)
        rstarts[i] = rstart
        rstart += ringinfo.numOfPixels
    end
    pixels = reduce(vcat, pixels) #FIXME:Is this the most efficient way?
    println("DistributedMap: I am task $c_rank of $c_size, I work on rings $rings of $(numOfRings(res)) \n")
    d_map.pixels = pixels
    d_map.rings = rings
    d_map.rstarts = rstarts
    d_map.info = make_subset_healpix_geom_info(map.resolution.nside, stride, rings)
    d_map.comm = comm
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
    - `strategy::Symbol`: Strategy to be used, e.g. pass `:RR` for "Round Robin".
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input map after having performed the "scattering".
"""
function MPI.Scatter!(
    in_map::HealpixMap{T,RingOrder,Array{T,1}},
    out_d_map::DistributedMap{T,I},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number, I <: Integer}

    if MPI.Comm_rank(comm) == root
        MPI.bcast(in_map, root, comm)
    else
        in_map = MPI.bcast(nothing, root, comm)
    end

    if strategy == :RR #Round Robin, can add more.
        ScatterMap_RR!(in_map, out_d_map, comm)
    end
    if clear
        in_map = nothing #free unnecessary copies of map
    end
end

function MPI.Scatter!(
    in_map::Nothing,
    out_d_map::DistributedMap{T,I},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number, I <: Integer}

    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input map on root task can NOT be `nothing`."))
    in_map = MPI.bcast(nothing, root, comm)

    if strategy == :RR #Round Robin, can add more.
        ScatterMap_RR!(in_map, out_d_map, comm)
    end
    if clear
        in_map = nothing #free unnecessary copies of map
    end
end


#######################################################################
#root task
function GatherMap_RR_root!(
    d_map::DistributedMap{T,I},
    map::HealpixMap{T,RingOrder,Array{T,1}},
    comm::MPI.Comm,
    root::Integer
    ) where {T <: Number, I <: Integer}

    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #each task can have at most root_nring + 1
    root_nring = length(d_map.rings)
    MPI.bcast(root_nring, root, comm)
    resolution = map.resolution
    MPI.bcast(resolution, root, comm)
    ringinfo = RingInfo(0, 0, 0, 0, 0)
    rings = d_map.rings
    @inbounds for ri in 1:root_nring+1
        if ri <= length(rings)
            getringinfo!(resolution, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[d_map.rstarts[ri]:d_map.rstarts[ri]+ringinfo.numOfPixels-1] #we select that ring's pixels
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
    comm::MPI.Comm,
    root::Integer
    ) where {T <: Number, I <: Integer}

    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #each task can have at most root_nring + 1
    root_nring = MPI.bcast(nothing, root, comm)
    resolution = MPI.bcast(nothing, root, comm)
    ringinfo = RingInfo(0, 0, 0, 0, 0)
    rings = d_map.rings
    @inbounds for ri in 1:root_nring+1
        if ri <= length(rings)
            getringinfo!(resolution, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[d_map.rstarts[ri]:d_map.rstarts[ri]+ringinfo.numOfPixels-1] #we select that ring's pixels
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
    - `strategy::Symbol`: Strategy to be used, e.g. pass `:RR` for "Round Robin".
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `DistributedMap` after having performed the "scattering".
"""
function MPI.Gather!(
    in_d_map::DistributedMap{T, I},
    out_map::HealpixMap{T,RingOrder,Array{T,1}},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number, I <: Integer}

    if strategy == :RR #Round Robin, can add more.
        if MPI.Comm_rank(comm) == root
            GatherMap_RR_root!(in_d_map, out_map, comm, root)
        else
            GatherMap_RR_rest!(in_d_map, comm, root)
        end
    end
    if clear
        in_d_map = nothing #free unnecessary copies of map
    end
end

#allows to pass nothing as output map on non-root tasks
function MPI.Gather!(
    in_d_map::DistributedMap{T, I},
    out_map::Nothing,
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number, I <: Integer}

    if strategy == :RR #Round Robin, can add more.
        GatherMap_RR_rest!(in_d_map, comm, root)
    end
    if clear
        in_d_map = nothing #free unnecessary copies of map
    end
end

##########################################################

function AllgatherMap_RR!(
    d_map::DistributedMap{T,I},
    map::HealpixMap{T,RingOrder,Array{T,1}},
    comm::MPI.Comm,
    root::Integer
    ) where {T <: Number, I <: Integer}

    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #each task can have at most root_nring + 1
    if crank == root
        root_nring = length(d_map.rings)
        MPI.bcast(root_nring, root, comm)
        resolution = map.resolution
        MPI.bcast(resolution, root, comm)
    else
        root_nring = MPI.bcast(nothing, root, comm)
        resolution = MPI.bcast(nothing, root, comm)
    end
    ringinfo = RingInfo(0, 0, 0, 0, 0)
    rings = d_map.rings
    @inbounds for ri in 1:root_nring+1
        if ri <= length(rings)
            getringinfo!(resolution, rings[ri], ringinfo; full=false) #it's a cheap computation
            local_count = ringinfo.numOfPixels
            local_displ = ringinfo.firstPixIdx - 1
            @views pixels = d_map.pixels[d_map.rstarts[ri]:d_map.rstarts[ri]+ringinfo.numOfPixels-1] #we select that ring's pixels
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
    - `strategy::Symbol`: Strategy to be used, e.g. pass `:RR` for "Round Robin".
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Allgather!(
    in_d_map::DistributedMap{T,I},
    out_map::HealpixMap{T,RingOrder,Array{T,1}},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where  {T <: Number, I <: Integer}

    if strategy == :RR #Round Robin, can add more.
        AllgatherMap_RR!(in_d_map, out_map, comm, root)
    end
    if clear
        in_d_map = nothing
    end
end
