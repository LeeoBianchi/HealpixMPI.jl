
#################### WAITING FOR NEW Healpix VERSION:
#=

function each_ell_idx(alm::Alm{Complex{T}}, m::Integer) where {T <: Number}
    (m <= alm.mmax) || throw(DomainError(m, "`m` is greater than mmax"))
    [i for i in almIndex(alm, m, m):almIndex(alm, alm.lmax, m)]
end

function each_ell_idx(alm::Alm{Complex{T}}, ms::AbstractArray{I, 1}) where {T <: Number, I <: Integer}
    reduce(vcat, [each_ell_idx(alm, m) for m in ms])
end
=#
#########################################################

""" struct AlmInfoMPI{I <: Integer}

Information describing an MPI-distributed subset of `Alm`, contained in a `DAlm`.

An `AlmInfoMPI` type contains:
- `comm`: MPI communicator used.
- `lmax`: the maximum value of ``ℓ``.
- `mmax`: the maximum value of ``m`` on the whole `Alm` set.
- `maxnm`: maximum value (over tasks) of nm (number of m values).
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
this).

A `SubAlm` type contains the following fields:

- `alm`: the array of harmonic coefficients
- `info`: an `AlmInfo` object describing the alm subset.

The `AlmInfo` contained in `info` must match exactly the characteristic of the Alm subset,
this can be constructed through the function `make_general_alm_info`, for instance.

"""
mutable struct DAlm{S<:Strategy, T<:Number, I<:Integer}
    alm::Vector{T}
    info::AlmInfoMPI{I}

    DAlm{S,T,I}(alm::Vector{T}, info::AlmInfoMPI{I}) where {S<:Strategy, T<:Number, I<:Integer} =
        new{S,T,I}(alm, info)
end

DAlm{S}(alm::Vector{T}, info::AlmInfoMPI{I}) where {S<:Strategy, T<:Number, I<:Integer} =
    DAlm{S,T,I}(alm, info)

#constructor with only comm
DAlm{S,T,I}(comm::MPI.Comm) where {S<:Strategy, T<:Number, I<:Integer} = DAlm{S}(Vector{T}(undef, 0), AlmInfoMPI{I}(comm))
DAlm{S}(comm::MPI.Comm) where {S<:Strategy} = DAlm{S, ComplexF64, Int64}(comm)

#empty constructors
DAlm{S,T,I}() where {S<:Strategy, T<:Number, I<:Integer} = DAlm{S}(Vector{T}(undef, 0), AlmInfoMPI{I}())
DAlm{S}() where {S<:Strategy} = DAlm{S, ComplexF64, Int64}()

# MPI Overloads:
## SCATTER
""" get_nm_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)

    Return number of m's on specified task in a Round Robin strategy
"""
function get_nm_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)::Integer
    (task_rank < c_size) || throw(DomainError(0, "$task_rank can not exceed communicator size"))
    (global_mmax + c_size - task_rank) ÷ c_size # =nm
end

""" get_mval_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)

    Return array of m values on specified task in a Round Robin strategy
"""
function get_mval_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)::Vector{Int}
    nm = get_nm_RR(global_mmax, task_rank, c_size)
    mval = Vector{Int}(undef, nm)
    @inbounds for i in 1:nm
        mval[i] = task_rank + (i - 1)*c_size #round robin strategy to divide the m's
    end
    mval
end

""" get_m_tasks_RR(mmax::Integer, c_size::Integer)

    Computes an array containing the task each m in the full range [0, `mmax`]
    is assigned to according to a Round Robin strategy, given the communicator size.
"""
function get_m_tasks_RR(mmax::Integer, c_size::Integer)::Vector{Int}
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

"""
    Internal function implementing a "Round Robin" strategy.

    Here the input alms are supposed to be on every task as a copy.
    The input Alm object is broadcasted by `MPI.Scatter!`.
"""
function ScatterAlm!(
    alm::Healpix.Alm{T,Array{T,1}},
    d_alm::DAlm{RR,T,I}
    ) where {T<:Number, I<:Integer}

    stride = 1
    c_rank = MPI.Comm_rank(d_alm.info.comm)
    c_size = MPI.Comm_size(d_alm.info.comm)
    mval = get_mval_RR(alm.mmax, c_rank, c_size)
    #if we have too many MPI tasks, some will be empty
    (!iszero(length(mval))) || throw(DomainError(0, "$c_rank-th task is empty."))
    d_alm.alm = @view alm.alm[Healpix.each_ell_idx(alm, mval)]
    d_alm.info.lmax = alm.lmax          #update info inplace
    d_alm.info.mmax = alm.mmax
    d_alm.info.maxnm = get_nm_RR(alm.mmax, 0, c_size) #due to RR the maxnm is the one on the task 0
    d_alm.info.mval = mval
    d_alm.info.mstart = make_mstart_complex(alm.lmax, stride, mval)
    println("DAlm: I am task $c_rank of $c_size, I work on $(length(mval)) m's of $(alm.mmax)")
end

import MPI: Scatter!, Gather!, Allgather!

"""
    Scatter!(in_alm::Alm{T,Array{T,1}}, out_d_alm::DAlm{T}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number, I<:Integer}
    Scatter!(in_alm::Nothing, out_d_alm::DAlm{T}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number, I<:Integer}
    Scatter!(in_alm, out_d_alm::DAlm{S,T,I}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number, I<:Integer}

    Distributes the `Alm` object passed in input on the `root` task overwriting the
    `DAlm` objects passed on each task, according to the specified strategy
    (by default ":RR" for Round Robin).

    As in the standard MPI function, the `in_alm` in input can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from the (potentially bulky) `Alm` object.

    # Arguments:
    - `in_alm::Alm{T,Array{T,1}}`: `Alm` object to distribute over the MPI tasks.
    - `out_d_alm::DAlm{T}`: output `DAlm` object.

    # Keywords:
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function Scatter!(
    in_alm::Healpix.Alm{T,Array{T,1}},
    out_d_alm::DAlm{S,T,I};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number, I<:Integer}

    comm = out_d_alm.info.comm
    if MPI.Comm_rank(comm) == root
        MPI.bcast(in_alm, root, comm)
    else
        in_alm = MPI.bcast(nothing, root, comm)
    end

    ScatterAlm!(in_alm, out_d_alm)

    if clear
        in_alm = nothing #free unnecessary copies of alm
    end
end

#non root node
function Scatter!(
    nothing,
    out_d_alm::DAlm{S,T,I};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number, I<:Integer}

    comm = out_d_alm.info.comm
    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input alm on root task can not be `nothing`."))
    in_alm = MPI.bcast(nothing, root, comm)

    ScatterAlm!(in_alm, out_d_alm)

    if clear
        in_alm = nothing #free unnecessary copies of alm
    end
end

function Scatter!(
    in_alm,
    out_d_alm::DAlm{S,T,I},
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number, I<:Integer}
    out_d_alm.info.comm = comm #overwrites the Comm in the d_alm
    MPI.Scatter!(in_alm, out_d_alm, root = root, clear = clear)
end

## GATHER
"""
    Internal function implementing a "Round Robin" strategy.

    Specifically relative to the root-task.
"""
function GatherAlm_root!(
    d_alm::DAlm{RR,T,I},
    alm::Healpix.Alm{T,Array{T,1}},
    root::Integer
    ) where {T<:Number, I<:Integer}

    comm = d_alm.info.comm
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

"""
    Internal function implementing a "Round Robin" strategy.

    Specifically relative to non root-tasks: no output is returned.
"""
function GatherAlm_rest!(
    d_alm::DAlm{RR,T,I},
    root::Integer
    ) where {T<:Number, I<:Integer}

    comm = d_alm.info.comm
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
    MPI.Gather!(in_d_alm::DAlm{S,T,I}, out_alm::Alm{T,Array{T,1}}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number, I<:Integer}
    MPI.Gather!(in_d_alm::DAlm{S,T,I}, out_alm::Nothing, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number, I<:Integer}

    Gathers the `DAlm` objects passed on each task overwriting the `Alm`
    object passed in input on the `root` task according to the specified `strategy`
    (by default `:RR` for Round Robin). Note that the strategy must match the one used
    to "scatter" the a_lm.

    As in the standard MPI function, the `out_alm` can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from
    the (potentially bulky) `DAlm` object.

    # Arguments:
    - `in_d_alm::DAlm{T}`: `DAlm` object to gather from the MPI tasks.
    - `out_d_alm::Alm{T,Array{T,1}}`: output `Alm` object.

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Gather!(
    in_d_alm::DAlm{S,T,I},
    out_alm::Healpix.Alm{T,Array{T,1}};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number, I<:Integer}

    if MPI.Comm_rank(in_d_alm.info.comm) == root
        GatherAlm_root!(in_d_alm, out_alm, root)
    else
        GatherAlm_rest!(in_d_alm, root)
    end
    if clear
        in_d_alm = nothing
    end
end

#allows non-root tasks to pass nothing as output
function MPI.Gather!(
    in_d_alm::DAlm{S,T,I},
    nothing;
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number, I<:Integer}

    (MPI.Comm_rank(in_d_alm.info.comm) != root)||throw(DomainError(0, "output alm on root task can not be `nothing`."))

    GatherAlm_rest!(in_d_alm, root) #on root out_alm cannot be nothing

    if clear
        in_d_alm = nothing
    end
end

## Allgather
"""
    Internal function implementing a "Round Robin" strategy.
"""
function AllgatherAlm!(
    d_alm::DAlm{RR,T,I},
    alm::Healpix.Alm{T,Array{T,1}},
    ) where {T<:Number, I<:Integer}

    comm = d_alm.info.comm
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
    MPI.Allgather!(in_d_alm::DAlm{S,T,I}, out_alm::Alm{T,Array{T,1}}; clear::Bool = false) where {S<:Strategy, T<:Number, I<:Integer}

    Gathers the `DAlm` objects passed on each task overwriting the `Alm`
    object passed in input on the `root` task according to the specified `strategy`
    (by default `:RR` for Round Robin). Note that the strategy must match the one used
    to "scatter" the a_lm.

    As in the standard MPI function, the `out_alm` can be `nothing` on non-root tasks,
    since it will be ignored anyway.

    If the keyword `clear` is set to `true` it frees the memory of each task from
    the (potentially bulky) `DAlm` object.

    # Arguments:
    - `in_d_alm::DAlm{T}`: `DAlm` object to gather from the MPI tasks.
    - `out_d_alm::Alm{T,Array{T,1}}`: output `Alm` object.

    # Keywords:
    - `strategy::Symbol`: Strategy to be used, by default `:RR` for "Round Robin".
    - `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function MPI.Allgather!(
    in_d_alm::DAlm{S,T,I},
    out_alm::Healpix.Alm{T,Array{T,1}};
    clear::Bool = false
    ) where {S<:Strategy, T<:Number, I<:Integer}

    AllgatherAlm!(in_d_alm, out_alm)

    if clear
        in_d_alm = nothing
    end
end

#########################################################################
"""
    ≃(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer}

Similarity operator, returns `true` if the two arguments have matching `info` objects.
"""
function ≃(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer}
    (&)((alm₁.info.comm == alm₂.info.comm),
        (alm₁.info.lmax == alm₂.info.lmax),
        (alm₁.info.mmax == alm₂.info.mmax),
        (alm₁.info.maxnm == alm₂.info.maxnm),
        (alm₁.info.mval == alm₂.info.mval),
        (alm₁.info.mstart == alm₂.info.mstart))
end

## DAlm Algebra
import Base.Threads

"""
    localdot(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer} -> Number

    Internal function for the MPI-parallel dot product.
    It performs a dot product LOCALLY on the current MPI task between the two
    `DAlm`s passed in input.

"""
function localdot(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer}
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
    dot(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer} -> Number

    MPI-parallel dot product between two `DAlm` object of matching size.
"""
function LinearAlgebra.dot(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer}
    comm = (alm₁.info.comm == alm₂.info.comm) ? alm₁.info.comm : throw(DomainError(0, "Communicators must match"))

    res = localdot(alm₁, alm₂)
    MPI.Allreduce(res, +, comm) #we sum together all the local results on each task
end

import Base: +, -, *, /

+(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer} =
    DAlm{S,T,I}(alm₁.alm .+ alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))
-(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer} =
    DAlm{S,T,I}(alm₁.alm .- alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))
*(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer} =
    DAlm{S,T,I}(alm₁.alm .* alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))
/(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer} =
    DAlm{S,T,I}(alm₁.alm ./ alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))

*(alm₁::DAlm{S,T,I}, c::Number) where {S<:Strategy, T<:Number, I<:Integer} =
    DAlm{S,T,I}(alm₁.alm .* c, alm₁.info)
*(c::Number, alm₁::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer} = alm₁ * c
/(alm₁::DAlm{S,T,I}, c::Number) where {S<:Strategy, T<:Number, I<:Integer} =
    DAlm{S,T,I}(alm₁.alm ./ c, alm₁.info)


"""
    almxfl!(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, I<:Integer, AA<:AbstractArray{T,1}}

Multiply IN-PLACE a subset of a_ℓm in the form of `DAlm` by a vector `fl`
representing an ℓ-dependent function.

# ARGUMENTS
- `alms::DAlm{S,T,I}`: The subset of spherical harmonics coefficients
- `fl::AbstractVector{T}`: The array giving the factor f_ℓ by which to multiply a_ℓm

"""
function Healpix.almxfl!(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, I<:Integer, AA<:AbstractArray{N,1}}

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
    almxfl(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, I<:Integer, AA<:AbstractArray{T,1}}

Multiply a subset of a_ℓm in the form of `DAlm` by a vector b_ℓ representing
an ℓ-dependent function, without changing the a_ℓm passed in input.

# ARGUMENTS
- `alm::DAlm{S,T,I}`: The array representing the spherical harmonics coefficients
- `fl::AbstractVector{T}`: The array giving the factor f_ℓ by which to multiply a_ℓm

#RETURNS
- `Alm{S,T}`: The result of a_ℓm * f_ℓ.
"""
function Healpix.almxfl(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, I<:Integer, AA<:AbstractArray{N,1}}
    alm_new = deepcopy(alm)
    Healpix.almxfl!(alm_new, fl)
    alm_new
end

""" *(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, I<:Integer, AA<:AbstractArray{T,1}}
    *(fl::AA, alm::DAlm{S,T,I}) where {S<:Strategy, T<:Number, I<:Integer, AA<:AbstractArray{T,1}}

    Perform the product of a `DAlm` object by a function of ℓ in a_ℓm space.
    Note: this consists in a shortcut of [`almxfl`](@ref), therefore a new `DAlm`
    object is returned.
"""
*(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, I<:Integer, AA<:AbstractArray{N,1}} = Healpix.almxfl(alm, fl)
*(fl::AA, alm::DAlm{S,T,I}) where {S<:Strategy, T<:Number, N<:Number, I<:Integer, AA<:AbstractArray{N,1}} = alm*fl

""" /(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, I<:Integer, AA<:AbstractArray{T,1}}

    Perform an element-wise division by a function of ℓ in a_ℓm space.
    A new `Alm` object is returned.
"""
/(alm::DAlm{S,T,I}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, I<:Integer, AA<:AbstractArray{N,1}} = Healpix.almxfl(alm, 1. ./ fl)
