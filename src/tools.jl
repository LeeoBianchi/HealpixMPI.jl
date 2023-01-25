using MPI #then remove, it's already in HealpixMPI.jl
using Healpix #then remove

include("map.jl")

function ScatterArray_RR(
    arr::AA,
    nside::Integer,
    comm::MPI.Comm
    ) where {T <: Real, AA <: AbstractArray{T,1}}
    local_rings = get_rindexes_RR(nside, MPI.Comm_rank(comm), MPI.Comm_size(comm))
    local_arr = Vector{Vector{Float64}}(undef, length(local_rings)) #vector of local 
    ring_info = RingInfo(0, 0, 0, 0, 0) #initialize ring info object
    res = Resolution(nside)
    i = 1
    for ring in local_rings
        getringinfo!(res, ring, ring_info; full=false)
        first_pix_idx = ring_info.firstPixIdx
        local_arr[i] = @view arr[first_pix_idx:(first_pix_idx + ring_info.numOfPixels - 1)]
    end
    local_arr = reduce(vcat, local_arr)
    return local_arr
end

function MPI.Scatter(
    arr::AA,
    nside::Integer,
    comm::MPI.Comm;
    strategy::Symbol = :RR,
    root::Integer = 0
    ) where {T <: Real, AA <: AbstractArray{T,1}}

    if MPI.Comm_rank(comm) == root
        MPI.bcast(arr, root, comm)
    else
        arr = MPI.bcast(nothing, root, comm)
    end

    if strategy == :RR #Round Robin, can add more.
        return ScatterArray_RR(arr, nside, comm)
    end
end

function MPI.Scatter(
    nothing,
    nside::Integer,
    comm::MPI.Comm;
    strategy::Symbol = :RR,
    root::Integer = 0
    )

    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input array on root task can NOT be `nothing`."))
    arr = MPI.bcast(nothing, root, comm)

    if strategy == :RR #Round Robin, can add more.
        return ScatterArray_RR(arr, nside, comm)
    end
end