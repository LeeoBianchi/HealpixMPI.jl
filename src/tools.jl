function ScatterArray_RR(
    arr::AA,
    nside::Integer,
    comm::MPI.Comm
    ) where {T <: Real, AA <: AbstractArray{T,1}}
    local_rings = get_rindexes_RR(nside, MPI.Comm_rank(comm), MPI.Comm_size(comm))
    local_arr = Vector{Vector{Float64}}(undef, length(local_rings)) #vector of local
    ring_info = Healpix.RingInfo(0, 0, 0, 0, 0) #initialize ring info object
    res = Healpix.Resolution(nside)
    i = 1
    for ring in local_rings
        Healpix.getringinfo!(res, ring, ring_info; full=false)
        first_pix_idx = ring_info.firstPixIdx
        local_arr[i] = arr[first_pix_idx:(first_pix_idx + ring_info.numOfPixels - 1)]
        i+=1
    end
    local_arr = reduce(vcat, local_arr)
    return local_arr
end

import MPI: Scatter

"""
    Scatter(arr::AA, nside::Integer, comm::MPI.Comm; strategy::Type = RR, root::Integer = 0) where {T <: Real, AA <: AbstractArray{T,1}}
    Scatter(nothing, nside::Integer, comm::MPI.Comm; strategy::Type = RR, root::Integer = 0)

    Distributes a map-space array (e.g. masks, diagonal noise matrices, etc.) passed in input on the `root` task,
    according to the specified strategy(e.g. pass ":RR" for Round Robin).

    As in the standard MPI function, the input `arr` can be `nothing` on non-root tasks, since it will be ignored anyway.

    # Arguments:
    - `arr::AA`: array to distribute over the MPI tasks.
    - `nside::Integer`: NSIDE parameter of the map we are referring to.
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
"""
function Scatter(
    arr::AA,
    nside::Integer,
    comm::MPI.Comm;
    strategy::Type = RR,
    root::Integer = 0
    ) where {T <: Real, AA <: AbstractArray{T,1}}

    if MPI.Comm_rank(comm) == root
        MPI.bcast(arr, root, comm)
    else
        arr = MPI.bcast(nothing, root, comm)
    end

    if strategy == RR #Round Robin, can add more.
        return ScatterArray_RR(arr, nside, comm)
    end
end

function Scatter(
    nothing,
    nside::Integer,
    comm::MPI.Comm;
    strategy::Type = RR,
    root::Integer = 0
    )

    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input array on root task can NOT be `nothing`."))
    arr = MPI.bcast(nothing, root, comm)

    if strategy == RR #Round Robin, can add more.
        return ScatterArray_RR(arr, nside, comm)
    end
end
