using MPI #then remove, it's already in HealpixMPI.jl
using Healpix

#################### WAITING FOR NEW Healpix VERSION:

function each_ell_idx(alm::Alm{Complex{T}}, m::Integer) where {T <: Number}
    (m <= alm.mmax) || throw(DomainError(m, "`m` is greater than mmax"))
    [i for i in almIndex(alm, m, m):almIndex(alm, alm.lmax, m)]
end

function each_ell_idx(alm::Alm{Complex{T}}, ms::AbstractArray{I, 1}) where {T <: Number, I <: Integer}
    reduce(vcat, [each_ell_idx(alm, m) for m in ms])
end

#########################################################

""" struct AlmInfoMPI{I <: Integer}

Information describing an MPI-distributed subset of `Alm`, contained in a `DistributedAlm`.

An `AlmInfoMPI` type contains:
- `lmax`: the maximum value of ``ℓ``.
- `mmax`: the maximum value of ``m`` on the whole `Alm` set.
- `maxnm`: maximum value (over tasks) of nm (number of m values).
- `comm`: MPI communicator used.
- `mval`: array of values of ``m`` contained in the subset.
- `mstart`: array hypothetical indexes of the harmonic coefficient with ℓ=0, m. #FIXME: for now 0-based

"""
mutable struct AlmInfoMPI{I <: Integer}
    #communicator
    comm::MPI.Comm

    #global info
    lmax::I
    mmax::I
    maxnm::I

    #local info
    mval::Vector{I}
    mstart::Vector{I}

    AlmInfoMPI{I}(lmax::I, mmax::I, maxnm::I, mval::Vector{I}, mstart::Vector{I}, comm::MPI.Comm) where {I <: Integer} =
        new{I}(comm, lmax, mmax, maxnm, mval, mstart)
end

AlmInfoMPI(lmax::I, mmax::I, maxnm::I, mval::Vector{I}, mstart::Vector{I}, comm::MPI.Comm) where {I <: Integer} =
    AlmInfoMPI{I}(lmax, mmax, maxnm, mval, mstart, comm)
AlmInfoMPI{I}(comm::MPI.Comm) where {I <: Integer} =
    AlmInfoMPI{I}(0, 0, 0, Vector{I}(undef,0), Vector{I}(undef,0), comm::MPI.Comm)
AlmInfoMPI{I}() where {I <: Integer} = AlmInfoMPI{I}(MPI.COMM_NULL)
AlmInfoMPI() = AlmInfoMPI{Int64}()


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
mutable struct DistributedAlm{T<:Number, I<:Integer}
    alm::Vector{T}
    info::AlmInfoMPI{I}

    DistributedAlm{T,I}(alm::Vector{T}, info::AlmInfoMPI{I}) where {T<:Number, I<:Integer} =
        new{T,I}(alm, info)
end

DistributedAlm(alm::Vector{T}, info::AlmInfoMPI{I}) where {T<:Number, I<:Integer} =
    DistributedAlm{T,I}(alm, info)

#constructor with only comm
DistributedAlm{T,I}(comm::MPI.Comm) where {T<:Number, I<:Integer} = DistributedAlm(Vector{T}(undef, 0), AlmInfoMPI{I}(comm))
DistributedAlm(comm::MPI.Comm) = DistributedAlm{ComplexF64, Int64}(comm)

#empty constructors
DistributedAlm{T,I}() where {T<:Number, I<:Integer} = DistributedAlm(Vector{T}(undef, 0), AlmInfoMPI{I}())
DistributedAlm() = DistributedAlm{ComplexF64, Int64}()

# MPI Overloads:
## SCATTER
"""
    Return number of m's on specified task in a Round Robin strategy
"""
function get_nm_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)
    (task_rank < c_size) || throw(DomainError(0, "$task_rank can not exceed communicator size"))
    (global_mmax + c_size - task_rank) ÷ c_size # =nm
end

"""
    Return array of m values on specified task in a Round Robin strategy
"""
function get_mval_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)
    nm = get_nm_RR(global_mmax, task_rank, c_size)
    mval = Vector{Int}(undef, nm)
    @inbounds for i in 1:nm
        mval[i] = task_rank + (i - 1)*c_size #round robin strategy to divide the m's
    end
    mval
end

"""
    Computes an array containing the task each m in the full range [0, `mmax`]
    is assigned to according to a Round Robin strategy, given the communicator size.
"""
function get_m_tasks_RR(mmax::Integer, c_size::Integer)
    res = Vector{Int}(undef, mmax+1)
    for m in 0:mmax
        res[m+1] = rem(m,c_size)
    end
    res
end

""" make_mstart_complex(lmax::Integer, stride::Integer, mval::AbstractArray{T}) where T <: Integer

    Computes the 1-based `mstart` array given any `mval` and `lmax` for `Alm` in complex
    representation.
"""
function make_mstart_complex(lmax::Integer, stride::Integer, mval::AbstractArray{T}) where T <: Integer
    idx = 1 #tracks the number of 'consumed' elements so far; need to correct by m
    mi = 1 #index of mstart array
    mstart = Vector{Int}(undef, length(mval))
    for m in mval
        mstart[mi] = stride * (idx - m) #fill mstart
        idx += lmax + 1 - m
        mi += 1
    end
    mstart
end

#here the input alms are supposed to be on every task as a copy
#the input Alm object is broadcasted by MPI.Scatter!
function ScatterAlm_RR!(
    alm::Alm{T,Array{T,1}},
    d_alm::DistributedAlm{T,I}
    ) where {T <: Number, I <: Integer}

    stride = 1
    c_rank = MPI.Comm_rank(d_alm.info.comm)
    c_size = MPI.Comm_size(d_alm.info.comm)
    mval = get_mval_RR(alm.mmax, c_rank, c_size)
    #if we have too many MPI tasks, some will be empty
    (!iszero(length(mval))) || throw(DomainError(0, "$c_rank-th task is empty."))
    d_alm.alm = @view alm.alm[each_ell_idx(alm, mval)]
    d_alm.info.lmax = alm.lmax          #update info inplace
    d_alm.info.mmax = alm.mmax
    d_alm.info.maxnm = get_nm_RR(alm.mmax, 0, c_size) #due to RR the maxnm is the one on the task 0
    d_alm.info.mval = mval
    d_alm.info.mstart = make_mstart_complex(alm.lmax, stride, mval)
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
    - `comm::MPI.Comm`: MPI communicator to use.

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Scatter!(
    in_alm::Alm{T,Array{T,1}},
    out_d_alm::DistributedAlm{T,I};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Number, I<:Integer}

    comm = out_d_alm.info.comm
    if MPI.Comm_rank(comm) == root
        MPI.bcast(in_alm, root, comm)
    else
        in_alm = MPI.bcast(nothing, root, comm)
    end

    if strategy == :RR #Round Robin, can add more.
        ScatterAlm_RR!(in_alm, out_d_alm)
    end
    if clear
        in_alm = nothing #free unnecessary copies of alm
    end
end

#non root node
function MPI.Scatter!(
    in_alm::Nothing,
    out_d_alm::DistributedAlm{T,I};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Number, I<:Integer}

    comm = out_d_alm.info.comm
    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input alm on root task can not be `nothing`."))
    in_alm = MPI.bcast(nothing, root, comm)

    if strategy == :RR #Round Robin, can add more.
        ScatterAlm_RR!(in_alm, out_d_alm)
    end
    if clear
        in_alm = nothing #free unnecessary copies of alm
    end
end

function MPI.Scatter!(
    in_alm,
    out_d_alm::DistributedAlm{T,I},
    comm::MPI.Comm;
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Number, I<:Integer}
    out_d_alm.info.comm = comm #overwrites the Comm in the d_alm
    MPI.Scatter!(in_alm, out_d_alm, strategy = strategy, root = root, clear = clear)
end

## GATHER

#root task
function GatherAlm_RR_root!(
    d_alm::DistributedAlm{T,I},
    alm::Alm{T,Array{T,1}},
    root::Integer
    ) where {T<:Number, I<:Integer}

    comm = d_alm.info.comm
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)
    #local quantities:
    lmax = d_alm.info.lmax
    local_mval = d_alm.info.mval
    local_nm = length(local_mval)
    local_mstart = d_alm.info.mstart
    displ_shift = 0
    @inbounds for mi in 1:d_alm.info.maxnm #loop over the "Robin's Rounds"
        if mi <= local_nm
            m = local_mval[mi]
            local_count = lmax - m + 1
            i1 = local_mstart[mi] + m
            i2 = i1 + lmax - m
            alm_chunk = @view d_alm.alm[i1:i2] #@view speeds up the slicing
        else
            local_count = 0
            alm_chunk = ComplexF64[]
        end
        counts = MPI.Gather(Int32(local_count), root, comm) #FIXME: can this communication be avoided?
        #MPI.Barrier(comm) #FIXME: is it necessary?
        outbuf = MPI.VBuffer(alm.alm, counts) #the output buffer points at the alms to overwrite
        outbuf.displs .+= displ_shift #we shift to the region in alm corresponding to the current round
        displ_shift += sum(counts) #we update the shift

        MPI.Gatherv!(alm_chunk, outbuf, root, comm) #gather the alms of this round
    end
end

#for NON-ROOT tasks: no output
function GatherAlm_RR_rest!(
    d_alm::DistributedAlm{T,I},
    root::Integer
    ) where {T<:Number, I<:Integer}

    comm = d_alm.info.comm
    crank = MPI.Comm_rank(d_alm.info.comm)
    csize = MPI.Comm_size(d_alm.info.comm)
    #local quantities:
    lmax = d_alm.info.lmax
    local_mval = d_alm.info.mval
    local_nm = length(local_mval)
    local_mstart = d_alm.info.mstart
    @inbounds for mi in 1:d_alm.info.maxnm #loop over the "Robin's Rounds"
        if mi <= local_nm
            m = local_mval[mi]
            local_count = lmax - m + 1
            i1 = local_mstart[mi] + m
            i2 = i1 + lmax - m
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

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Gather!(
    in_d_alm::DistributedAlm{T,I},
    out_alm::Alm{T,Array{T,1}};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Number, I<:Integer}

    if strategy == :RR #Round Robin, can add more.
        if MPI.Comm_rank(in_d_alm.info.comm) == root
            GatherAlm_RR_root!(in_d_alm, out_alm, root)
        else
            GatherAlm_RR_rest!(in_d_alm, root)
        end
    end
    if clear
        in_d_alm = nothing
    end
end

#allows non-root tasks to pass nothing as output
function MPI.Gather!(
    in_d_alm::DistributedAlm{T,I},
    out_alm::Nothing;
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Number, I<:Integer}

    (MPI.Comm_rank(in_d_alm.info.comm) != root)||throw(DomainError(0, "output alm on root task can not be `nothing`."))

    if strategy == :RR #Round Robin, can add more.
        GatherAlm_RR_rest!(in_d_alm, root) #on root out_alm cannot be nothing
    end
    if clear
        in_d_alm = nothing
    end
end

## Allgather

function AllgatherAlm_RR!(
    d_alm::DistributedAlm{T,I},
    alm::Alm{T,Array{T,1}},
    root::Integer
    ) where {T<:Number, I<:Integer}

    comm = d_alm.info.comm
    crank = MPI.Comm_rank(comm)
    csize = MPI.Comm_size(comm)

    #local quantities:
    lmax = d_alm.info.lmax
    local_mval = d_alm.info.mval
    local_nm = length(local_mval)
    local_mstart = d_alm.info.mstart
    displ_shift = 0
    @inbounds for mi in 1:d_alm.info.maxnm #loop over the "Robin's Rounds"
        if mi <= local_nm
            m = local_mval[mi]
            local_count = lmax - m + 1
            i1 = local_mstart[mi] + m
            i2 = i1 + lmax - m
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

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Allgather!(
    in_d_alm::DistributedAlm{T,I},
    out_alm::Alm{T,Array{T,1}};
    strategy::Symbol = :RR,
    root::Integer = 0,
    clear::Bool = false
    ) where {T<:Number, I<:Integer}

    if strategy == :RR #Round Robin, can add more.
        AllgatherAlm_RR!(in_d_alm, out_alm, root)
    end
    if clear
        in_d_alm = nothing
    end
end

## DistributedAlm Algebra
import LinearAlgebra: dot
import Base.Threads

"""
    localdot(alm₁::DistributedAlm{Complex{T}}, alm₂::DistributedAlm{Complex{T}}) where {T <: Number} -> Number

    Internal function for the MPI-parallel dot product.
    It performs a dot product LOCALLY on the current MPI task between the two
    `DistributedAlm`s passed in input.

"""
function localdot(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    lmax = (alm₁.info.lmax == alm₁.info.lmax) ? alm₁.info.lmax : throw(DomainError(1, "lmax must match"))
    mval = (alm₁.info.mval == alm₂.info.mval) ? alm₁.info.mval : throw(DomainError(2, "mval must match"))
    mstart = (alm₁.info.mstart == alm₂.info.mstart) ? alm₁.info.mstart : throw(DomainError(3, "mstarts must match"))
    nm = length(mval)
    res_m0 = 0
    res_rest = 0
    @inbounds for mi in 1:nm #maybe run in parallel with Threads.@threads
        m = mval[mi]
        i1 = mstart[mi] + m
        i2 = i1 + lmax - m #this gives index range for each ell, for given m
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
function dot(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    comm = (alm₁.info.comm == alm₂.info.comm) ? alm₁.info.comm : throw(DomainError(0, "Communicators must match"))

    res = localdot(alm₁, alm₂)
    MPI.Barrier(comm) #FIXME: necessary??
    println("task $(MPI.Comm_rank(comm)), dot = $res")
    MPI.Allreduce(res, +, comm) #we sum together all the local results on each task
end

import Base: +, -, *, /

""" +(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}

    Perform the element-wise sum in a_ℓm space.
    A new `Alm` object is returned.
"""
function +(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    (alm₁.info == alm₂.info) || throw(DomainError("infos not matching"))

    res_alm = deepcopy(alm₁)

     @inbounds for i in 1:length(res_alm.alm)
        res_alm.alm[i] = alm₁.alm[i] + alm₂.alm[i]
    end
    res_alm
end

""" -(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}

    Perform the element-wise subtraction in a_ℓm space.
    A new `Alm` object is returned.
"""
function -(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    (alm₁.info == alm₂.info) || throw(DomainError("infos not matching"))

    res_alm = deepcopy(alm₁)

     @inbounds for i in 1:length(res_alm.alm)
        res_alm.alm[i] = alm₁.alm[i] - alm₂.alm[i]
    end
    res_alm
end

""" *(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}

    Perform the element-wise product in a_ℓm space.
    A new `Alm` object is returned.
"""
function *(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    (alm₁.info == alm₂.info) || throw(DomainError("infos not matching"))
    lmax = alm₁.info.lmax

    #first part of the alm arrays, where for m=0 whe have alm^R_l,0 = alm^C_l,0,
    #and thus only real values are interesting (imag. should be 0)
    res_alm = deepcopy(alm₁)
    @inbounds for i in 1:lmax+1
        res_alm.alm[i] = real(alm₁.alm[i]) * real(alm₂.alm[i])
    end
    #we then compute the rest, where alm^R_l,m = √2 Re{alm^C_l,m}, alm^R_l,-m = √2Im{alm^C_l,m}
    @inbounds for i in lmax+2:length(alm₁.alm)
        res_alm.alm[i] = alm₁.alm[i] * alm₂.alm[i]
    end
    res_alm
end

"""
    almxfl!(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}}

Multiply IN-PLACE a subset of a_ℓm in the form of `DistributedAlm` by a vector `fl`
representing an ℓ-dependent function.

# ARGUMENTS
- `alms::DistributedAlm{Complex{T},I}`: The subset of spherical harmonics coefficients
- `fl::AbstractVector{T}`: The array giving the factor f_ℓ by which to multiply a_ℓm

""" #FIXME: then import it from Healpix.jl and overlad it when it will be available
function almxfl!(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}}

    lmax = alm.info.lmax
    mval = alm.info.mval
    fl_size = length(fl)

    if lmax + 1 > fl_size
        fl = [fl; zeros(lmax + 1 - fl_size)]
    end
    i = 1
    @inbounds for m in mval
        for l = m:lmax
            alm.alm[i] = alm.alm[i]*fl[l+1]
            i += 1
        end
    end
end

"""
    almxfl(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}}

Multiply a subset of a_ℓm in the form of `DistributedAlm` by a vector b_ℓ representing
an ℓ-dependent function, without changing the a_ℓm passed in input.

# ARGUMENTS
- `alm::Alm{Complex{T}}`: The array representing the spherical harmonics coefficients
- `fl::AbstractVector{T}`: The array giving the factor f_ℓ by which to multiply a_ℓm

#RETURNS
- `Alm{Complex{T}}`: The result of a_ℓm * f_ℓ.
""" #FIXME: idem..
function almxfl(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}}
    alm_new = deepcopy(alm)
    almxfl!(alm_new, fl)
    alm_new
end

""" *(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}}

    Perform the product of a `DistributedAlm` object by a function of ℓ in a_ℓm space.
    Note: this consists in a shortcut of [`almxfl`](@ref), therefore a new `DistributedAlm`
    object is returned.
"""
*(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}} = almxfl(alm, fl)

""" *(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}

    Perform the element-wise product of an `DistributedAlm` object by a constant in a_ℓm space.
"""
function *(alm₁::DistributedAlm{Complex{T},I}, c::Number) where {T<:Real, I<:Integer}
    res_alm = deepcopy(alm₁)

    @inbounds for i in 1:length(alm₁.alm)
        res_alm.alm[i] = alm₁.alm[i] * c
    end
    res_alm
end

""" /(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}

    Perform the element-wise product in a_ℓm space.
    A new `Alm` object is returned.
"""
function /(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    (alm₁.info == alm₂.info) || throw(DomainError("infos not matching"))
    lmax = alm₁.info.lmax

    #first part of the alm arrays, where for m=0 whe have alm^R_l,0 = alm^C_l,0,
    #and thus only real values are interesting (imag. should be 0)
    res_alm = deepcopy(alm₁)
    @inbounds for i in 1:lmax+1
        res_alm.alm[i] = real(alm₁.alm[i]) / real(alm₂.alm[i])
    end
    #we then compute the rest, where alm^R_l,m = √2 Re{alm^C_l,m}, alm^R_l,-m = √2Im{alm^C_l,m}
    @inbounds for i in lmax+2:length(alm₁.alm)
        res_alm.alm[i] = alm₁.alm[i] / alm₂.alm[i]
    end
    res_alm
end

""" /(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}}

    Perform an element-wise division by a function of ℓ in a_ℓm space.
    A new `Alm` object is returned.
"""
/(alm::DistributedAlm{Complex{T},I}, fl::AA) where {T<:Real, I<:Integer, AA<:AbstractArray{T,1}} = almxfl(alm, 1. ./ fl)

""" /(alm₁::DistributedAlm{Complex{T},I}, c::Number) where {T<:Real, I<:Integer}

    Perform an element-wise division by a constant in a_ℓm space.
    A new `Alm` object is returned.
"""
function /(alm₁::DistributedAlm{Complex{T},I}, c::Number) where {T<:Real, I<:Integer}
    res_alm = deepcopy(alm₁)

    @inbounds for i in eachindex(alm₁)
        res_alm.alm[i] = alm₁.alm[i] / c
    end
    res_alm
end
