include("alminfo.jl")

""" An MPI-distributed subset of harmonic coefficients a_ℓm, referring only to certain values of m.

The type `T` is used for the value of each harmonic coefficient, and
it must be a `Number` (one should however only use complex types for
this). The type `AA` is used to store the array of coefficients; a
typical choice is `Vector`.

A `SubAlm` type contains the following fields:

- `alm`: the array of harmonic coefficients
- `info`: an `AlmInfo` object describing the alm subset.
- `comm`: `MPI.Comm` communicator used to distribute the a_lm.

The `AlmInfo` contained in `info` must match exactly the characteristic of the Alm subset,
this can be constructed through the function `make_general_alm_info`, for instance.

"""
mutable struct DistributedAlm{T <: Number}
    alm::Vector{T}
    info::AlmInfo
    comm::MPI.Comm

    DistributedAlm{T}(alm::Vector{T}, info::AlmInfo, comm::MPI.Comm) where {T <: Number} =
        new{T}(alm, info, comm)
end

DistributedAlm(alm::Vector{T}, info::AlmInfo, comm::MPI.Comm) where {T <: Number} =
    DistributedAlm{T}(alm, info, comm)

#constructor that assumes Healpix.jl-style alm and stride=1
DistributedAlm(alm::Vector{T}, mval::Vector{I}, lmax::Integer, comm::MPI.Comm) where {T <: Number, I <: Integer} =
    DistributedAlm{T}(alm, make_mmajor_complex_alm_info(lmax, 1, mval), comm)

#empty constructors
DistributedAlm{T}() where {T <: Number} = DistributedAlm(Vector{T}(undef, 0), Int[], 0, MPI.COMM_WORLD)
DistributedAlm() = DistributedAlm{ComplexF64}()

# MPI Overloads:
## SCATTER

#here the input alms are supposed to be on every task as a copy
#the input Alm object is broadcasted by MPI.Scatter!
function ScatterAlm_RR!(
    alm::Alm{T,Array{T,1}},
    d_alm::DistributedAlm{T},
    comm::MPI.Comm
    ) where {T <: Number}

    stride = 1
    c_rank = MPI.Comm_rank(comm)
    c_size = MPI.Comm_size(comm)
    nm = (alm.mmax + c_size - c_rank) ÷ c_size #number local m's
    mval = Vector{Int}(undef, nm)

    #if we have too many MPI tasks, some will be empty
    (!iszero(nm)) || throw(DomainError(0, "$c_rank-th MPI task is empty."))

    mval[1] = c_rank
    @inbounds for i in 2:nm
        mval[i] = c_rank + (i - 1)*c_size #round robin strategy to divide the m's
    end
    d_alm.alm = @view alm.alm[each_ell_idx(alm, mval)]
    d_alm.info = make_mmajor_complex_alm_info(alm.lmax, stride, mval) #updates info inplace
    d_alm.comm = comm
    println("DistributedAlm: I am task $c_rank of $c_size, I work on m's $mval of $(alm.mmax) \n")
end

"""
    MPI.Scatter!(in_alm::Alm{T,Array{T,1}}, out_d_alm::DistributedAlm{T}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number}
    MPI.Scatter!(in_alm::Nothing, out_d_alm::DistributedAlm{T}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number}

    Distributes the `Alm` object passed in input on the `root` task overwriting the
    `DistributedAlm` objects passed on each task, according to the specified strategy
    (e.g. pass ":RR" for Round Robin).

    As in the standard MPI function, the `in_alm` in input can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from the (potentially bulky) `Alm` object.

    # Arguments:
    - `in_alm::Alm{T,Array{T,1}}`: `Alm` object to distribute over the MPI tasks.
    - `out_d_alm::DistributedAlm{T}`: output `DistributedAlm` object.
    - `strategy::Symbol`: Strategy to be used, e.g. pass `:RR` for "Round Robin".
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Scatter!(
    in_alm::Alm{T,Array{T,1}},
    out_d_alm::DistributedAlm{T},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number}

    if MPI.Comm_rank(comm) == root
        MPI.bcast(in_alm, root, comm)
    else
        in_alm = MPI.bcast(nothing, root, comm)
    end

    if strategy == :RR #Round Robin, can add more.
        ScatterAlm_RR!(in_alm, out_d_alm, comm)
    end
    if clear
        in_alm = nothing #free unnecessary copies of alm
    end
end

#non root node
function MPI.Scatter!(
    in_alm::Nothing,
    out_d_alm::DistributedAlm{T},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number}

    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input alm on root task can not be `nothing`."))
    in_alm = MPI.bcast(nothing, root, comm)

    if strategy == :RR #Round Robin, can add more.
        ScatterAlm_RR!(in_alm, out_d_alm, comm)
    end
    if clear
        in_alm = nothing #free unnecessary copies of alm
    end
end

## GATHER

#root task
function GatherAlm_RR_root!(
    d_alm::DistributedAlm{T},
    alm::Alm{T,Array{T,1}},
    comm::MPI.Comm,
    root::Integer
    ) where {T <: Number}

    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #each task can have at most the same number of m's as the root
    root_nm = d_alm.info.nm
    MPI.bcast(root_nm, root, comm)
    #local quantities:
    local_nm = d_alm.info.nm
    local_lmax = d_alm.info.lmax
    local_mval = d_alm.info.mval
    local_mvstart = d_alm.info.mvstart
    displ_shift = 0
    @inbounds for mi in 1:root_nm #loop over the "Robin's Rounds"
        if mi <= local_nm
            m = local_mval[mi]
            local_count = local_lmax - m + 1
            i1 = local_mvstart[mi] + 1 + m  #FIXME: embed this in a getindex function with @views
            i2 = i1 + local_lmax - m
            alm_chunk = @view d_alm.alm[i1:i2] #@view speeds up the slicing
        else
            local_count = 0
            alm_chunk = ComplexF64[]
        end
        counts = MPI.Gather(Int32(local_count), root, comm)
        #MPI.Barrier(comm) #FIXME: is it necessary?
        outbuf = MPI.VBuffer(alm.alm, counts) #the output buffer points at the alms to overwrite
        outbuf.displs .+= displ_shift #we shift to the region in alm corresponding to the current round
        displ_shift += sum(counts) #we update the shift

        MPI.Gatherv!(alm_chunk, outbuf, root, comm) #gather the alms of this round
    end
end

#for NON-ROOT tasks: no output
function GatherAlm_RR_rest!(
    d_alm::DistributedAlm{T},
    comm::MPI.Comm,
    root::Integer
    ) where {T <: Number}

    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #each task can have at most the same number of m's as the root
    root_nm = MPI.bcast(nothing, root, comm)
    #local quantities:
    local_nm = d_alm.info.nm
    local_lmax = d_alm.info.lmax
    local_mval = d_alm.info.mval
    local_mvstart = d_alm.info.mvstart
    @inbounds for mi in 1:root_nm #loop over the "Robin's Rounds"
        if mi <= local_nm
            m = local_mval[mi]
            local_count = local_lmax - m + 1
            i1 = local_mvstart[mi] + 1 + m  #FIXME: embed this in a getindex function (?)
            i2 = i1 + local_lmax - m
            alm_chunk = @view d_alm.alm[i1:i2] #get the chunk of alm to send to root
        else
            local_count = 0
            alm_chunk = ComplexF64[]
        end
        MPI.Gather(Int32(local_count), root, comm)
        #MPI.Barrier(comm) #FIXME: is it necessary?
        MPI.Gatherv!(alm_chunk, nothing, root, comm) #gather the alms of this round
    end
end

"""
    MPI.Gather!(in_d_alm::DistributedAlm{T}, out_alm::Alm{T,Array{T,1}}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number}
    MPI.Gather!(in_d_alm::DistributedAlm{T}, out_alm::Nothing, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number}

    Gathers the `DistributedAlm` objects passed on each task overwriting the `Alm`
    object passed in input on the `root` task according to the specified `strategy`
    (e.g. pass `:RR` for Round Robin). Note that the strategy must match the one used
    to "scatter" the a_lm.

    As in the standard MPI function, the `out_alm` can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from
    the (potentially bulky) `DistributedAlm` object.

    # Arguments:
    - `in_d_alm::DistributedAlm{T}`: `DistributedAlm` object to gather from the MPI tasks.
    - `out_d_alm::Alm{T,Array{T,1}}`: output `Alm` object.
    - `strategy::Symbol`: Strategy to be used, e.g. pass `:RR` for "Round Robin".
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Gather!(
    in_d_alm::DistributedAlm{T},
    out_alm::Alm{T,Array{T,1}},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number}

    if strategy == :RR #Round Robin, can add more.
        if MPI.Comm_rank(comm) == root
            GatherAlm_RR_root!(in_d_alm, out_alm, comm, root)
        else
            GatherAlm_RR_rest!(in_d_alm, comm, root)
        end
    end
    if clear
        in_d_alm = nothing
    end
end

#allows non-root tasks to pass nothing as output
function MPI.Gather!(
    in_d_alm::DistributedAlm{T},
    out_alm::Nothing,
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number}

    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "output alm on root task can not be `nothing`."))

    if strategy == :RR #Round Robin, can add more.
        GatherAlm_RR_rest!(in_d_alm, comm, root) #on root out_alm cannot be nothing
    end
    if clear
        in_d_alm = nothing
    end
end

## Allgather

function AllgatherAlm_RR!(
    d_alm::DistributedAlm{T},
    alm::Alm{T,Array{T,1}},
    comm::MPI.Comm,
    root::Integer
    ) where {T <: Number}

    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #each task can have at most the same number of m's as the root
    if crank == root
        root_nm = d_alm.info.nm
        MPI.bcast(root_nm, root, comm)
    else
        root_nm = MPI.bcast(nothing, root, comm)
    end
    #local quantities:
    local_nm = d_alm.info.nm
    local_lmax = d_alm.info.lmax
    local_mval = d_alm.info.mval
    local_mvstart = d_alm.info.mvstart
    displ_shift = 0
    @inbounds for mi in 1:root_nm #loop over the "Robin's Rounds"
        if mi <= local_nm
            m = local_mval[mi]
            local_count = local_lmax - m + 1
            i1 = local_mvstart[mi] + 1 + m  #FIXME: embed this in a getindex function with @views
            i2 = i1 + local_lmax - m
            alm_chunk = @view d_alm.alm[i1:i2] #@view speeds up the slicing
        else
            local_count = 0
            alm_chunk = ComplexF64[]
        end
        counts = MPI.Allgather(Int32(local_count), comm)
        #MPI.Barrier(comm) #FIXME: is it necessary?
        outbuf = MPI.VBuffer(alm.alm, counts) #the output buffer points at the alms to overwrite
        outbuf.displs .+= displ_shift #we shift to the region in alm corresponding to the current round
        displ_shift += sum(counts) #we update the shift

        MPI.Allgatherv!(alm_chunk, outbuf, comm) #gather the alms of this round
    end
end

"""
    MPI.Allgather!(in_d_alm::DistributedAlm{T}, out_alm::Alm{T,Array{T,1}}, strategy::Symbol, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {T <: Number}

    Gathers the `DistributedAlm` objects passed on each task overwriting the `Alm`
    object passed in input on the `root` task according to the specified `strategy`
    (e.g. pass `:RR` for Round Robin). Note that the strategy must match the one used
    to "scatter" the a_lm.

    As in the standard MPI function, the `out_alm` can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from
    the (potentially bulky) `DistributedAlm` object.

    # Arguments:
    - `in_d_alm::DistributedAlm{T}`: `DistributedAlm` object to gather from the MPI tasks.
    - `out_d_alm::Alm{T,Array{T,1}}`: output `Alm` object.
    - `strategy::Symbol`: Strategy to be used, e.g. pass `:RR` for "Round Robin".
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Allgather!(
    in_d_alm::DistributedAlm{T},
    out_alm::Alm{T,Array{T,1}},
    strategy::Symbol,
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {T <: Number}

    if strategy == :RR #Round Robin, can add more.
        AllgatherAlm_RR!(in_d_alm, out_alm, comm, root)
    end
    if clear
        in_d_alm = nothing
    end
end

## DistributedAlm Algebra
import LinearAlgebra: dot

"""
    localdot(alm₁::DistributedAlm{Complex{T}}, alm₂::DistributedAlm{Complex{T}}) where {T <: Number} -> Number

    Internal function for the MPI-parallel dot product.
    It performs a dot product LOCALLY on the current MPI task between the two
    `DistributedAlm`s passed in input.

"""
function localdot(alm₁::DistributedAlm{Complex{T}}, alm₂::DistributedAlm{Complex{T}}) where {T <: Number}
    lmax = (alm₁.info.lmax == alm₁.info.lmax) ? alm₁.info.lmax : throw(DomainError(1, "lmax must match"))
    mval = (alm₁.info.mval == alm₂.info.mval) ? alm₁.info.mval : throw(DomainError(2, "mval must match"))
    mvstart = (alm₁.info.mvstart == alm₂.info.mvstart) ? alm₁.info.mvstart : throw(DomainError(3, "mvstarts must match"))
    nm = alm₁.info.nm
    res_m0 = 0
    res_rest = 0
    @inbounds for mi in 1:nm #maybe run in parallel with JuliaThreads
        m = mval[mi]
        i1 = mvstart[mi] + 1 + m #+1 because Julia is 1-based
        i2 = i1 + lmax - m #this gives index range for each ell for given m
        if (m == 0)
            @inbounds for i in i1:i2
                res_m0 += real(alm₁.alm[i]) * real(alm₂.alm[i]) #if m=0 we have no imag part
            end
        else
            @inbounds for i in i1:i2
                res_rest += real(alm₁.alm[i]) * real(alm₂.alm[i]) + imag(alm₁.alm[i]) * imag(alm₂.alm[i])
            end
        end
    end
    return res_m0 + 2*res_rest
end

"""
    dot(alm₁::DistributedAlm{Complex{T}}, alm₂::DistributedAlm{Complex{T}}) where {T <: Number} -> Number

    MPI-parallel dot product between two `DistributedAlm` object of matching size.
"""
function dot(alm₁::DistributedAlm{Complex{T}}, alm₂::DistributedAlm{Complex{T}}) where {T <: Number}
    comm = (alm₁.comm == alm₂.comm) ? alm₁.comm : throw(DomainError(0, "Communicators must match"))

    res = localdot(alm₁::DistributedAlm{Complex{T}}, alm₂::DistributedAlm{Complex{T}})
    MPI.Barrier(comm)
    print("task $(MPI.Comm_rank(comm)), dot = $res")
    MPI.Allreduce(res, +, comm) #we sum together all the local results on each task
end
