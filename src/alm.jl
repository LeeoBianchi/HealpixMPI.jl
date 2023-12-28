
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

"""
	struct AlmInfoMPI

Information describing an MPI-distributed subset of `Alm`, contained in a `DAlm`.

An `AlmInfoMPI` type contains:
- `comm`: MPI communicator used.
- `lmax`: the maximum value of ``ℓ``.
- `mmax`: the maximum value of ``m`` on the whole `Alm` set.
- `maxnm`: maximum value (over tasks) of nm (number of m values).
- `mval`: array of values of ``m`` contained in the subset.
- `mstart`: array hypothetical indexes of the harmonic coefficient with ℓ=0, m. #FIXME: for now 0-based

"""
mutable struct AlmInfoMPI
    #communicator
    comm::MPI.Comm

    #global info
    lmax::Int
    mmax::Int
    maxnm::Int

    #local info
    mval::Vector{Int}
    mstart::Vector{Int}

    AlmInfoMPI(lmax::I, mmax::I, maxnm::I, mval::Vector{I}, mstart::Vector{I}, comm::MPI.Comm) where {I <: Integer} =
        new(comm, lmax, mmax, maxnm, mval, mstart)
end

AlmInfoMPI(comm::MPI.Comm) =
    AlmInfoMPI(0, 0, 0, Vector{Int}(undef,0), Vector{Int}(undef,0), comm::MPI.Comm)
AlmInfoMPI() = AlmInfoMPI(MPI.COMM_NULL)

"""
    Abstract type to allow multiple dispatch.
"""
abstract type AbstractDAlm end

"""
	struct DAlm{S<:Strategy, T<:Number}

An MPI-distributed subset of harmonic coefficients a_ℓm, referring only to certain values of m.

The type `T` is used for the value of each harmonic coefficient, and
it must be a `Number` (one should however only use complex types for
this).

A `SubAlm` type contains the following fields:

- `alm::Matrix{T}`: the array of harmonic coefficients, of dimensions `(nalm, ncomp)`.
- `info::AlmInfoMPI{I}`: an `AlmInfo` object describing the alm subset.

`ncomp` can be greater than one for supporting polarized shts.
The `AlmInfo` contained in `info` must match exactly the characteristic of the Alm subset,
this can be constructed through the function `make_general_alm_info`, for instance.

"""
mutable struct DAlm{S<:Strategy, T<:Number} <: AbstractDAlm
    alm::Matrix{T} #alias for Array{T,2}
    info::AlmInfoMPI

    DAlm{S,T}(alm::Matrix{T}, info::AlmInfoMPI) where {S<:Strategy, T<:Number} =
        new{S,T}(alm, info)
end

DAlm{S}(alm::Matrix{T}, info::AlmInfoMPI) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(alm, info)

#spin-0 constructor
DAlm{S}(alm::Vector{T}, info::AlmInfoMPI) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(reshape(alm, length(alm), 1), info)

#constructor with only comm
DAlm{S,T}(comm::MPI.Comm) where {S<:Strategy, T<:Number} = DAlm{S}(Matrix{T}(undef, 0, 0), AlmInfoMPI(comm))
DAlm{S}(comm::MPI.Comm) where {S<:Strategy} = DAlm{S, ComplexF64}(comm)

#empty constructors
DAlm{S,T}() where {S<:Strategy, T<:Number} = DAlm{S}(Matrix{T}(undef, 0, 0), AlmInfoMPI())
DAlm{S}() where {S<:Strategy} = DAlm{S, ComplexF64}()

#Overload of size operator
Base.size(alm::DAlm{S,T}) where {S,T} = size(m.alm)

# MPI Overloads:
## SCATTER
"""
    get_nm_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)

Return number of m's on specified task in a Round Robin strategy
"""
function get_nm_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)::Integer
    (task_rank < c_size) || throw(DomainError(0, "$task_rank can not exceed communicator size"))
    (global_mmax + c_size - task_rank) ÷ c_size # =nm
end

"""
    get_mval_RR(global_mmax::Integer, task_rank::Integer, c_size::Integer)

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

"""
	get_m_tasks_RR(mmax::Integer, c_size::Integer)

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

"""
	make_mstart_complex(lmax::Integer, stride::Integer, mval::AbstractArray{T}) where T <: Integer

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
    d_alm::DAlm{RR,T}
    ) where {T<:Number}
    stride = 1
    c_rank = MPI.Comm_rank(d_alm.info.comm)
    c_size = MPI.Comm_size(d_alm.info.comm)
    mval = get_mval_RR(alm.mmax, c_rank, c_size)
    #if we have too many MPI tasks, some will be empty
    (!iszero(length(mval))) || throw(DomainError(0, "$c_rank-th task is empty."))
    d_alm.alm = reshape(alm.alm[Healpix.each_ell_idx(alm, mval)], :, 1)
    d_alm.info.lmax = alm.lmax          #update info inplace
    d_alm.info.mmax = alm.mmax
    d_alm.info.maxnm = get_nm_RR(alm.mmax, 0, c_size) #due to RR the maxnm is the one on the task 0
    d_alm.info.mval = mval
    d_alm.info.mstart = make_mstart_complex(alm.lmax, stride, mval)
    println("DAlm: I am task $c_rank of $c_size, I work on $(length(mval)) m's of $(alm.mmax)")
end

#polarization components only
function ScatterAlm!(
    alms::AbstractArray{Healpix.Alm{T,Array{T,1}}, 1},
    d_alm::DAlm{RR,T}
    ) where {T<:Number}

    ScatterAlm!(alms[1], d_alm) #first we call the scalar function to build the info in d_alms          #E
    d_alm.alm = cat(d_alm.alm, alms[2].alm[Healpix.each_ell_idx(alms[2], d_alm.info.mval)], dims = 2)   #B
end

import MPI: Scatter!, Gather!, Allgather!

"""
    Scatter!(in_alm::Union{Healpix.Alm{T,Array{T,1}}, Vector{Healpix.Alm{T,Array{T,1}}}}, out_d_alm::DAlm{T}; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Scatter!(in_alm::Vector{Healpix.Alm{T,Array{T,1}}}, out_d_alm::DAlm{T}, out_d_pol_alm::DAlm{T}; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Scatter!(::Nothing, out_d_alm::DAlm{T}; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Scatter!(::Nothing, out_d_alm::DAlm{T}, out_d_pol_alm::DAlm{T}; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Scatter!(in_alm, out_d_alm::DAlm{S,T}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Scatter!(in_alm, out_d_alm::DAlm{S,T}, out_d_alm::DAlm{S,T}, comm::MPI.Comm; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}

Distributes the `Alm` object passed in input on the `root` task overwriting the
`DAlm` objects passed on each task, according to the specified strategy.

As in the standard MPI function, the `in_alm` in input can be `nothing` on non-root tasks,
since it will be ignored anyway.

To distribute a set of Alms representing a POLARIZED field there are 2 options:
- Pass in input a `Vector{Healpix.Alm}` with only E and B components and one output `DAlm` object which will contain both.
- Pass in input a `Vector{Healpix.Alm}` with T, E and B components and two output `DAlm` objects which will contain T and E & B respectively.
This is so that the resulting `DAlm` objects can be directly passed to the sht functions which only accept in input the intensity component for a scalar transform and two polarization components for a spinned transform.

If the keyword `clear` is set to `true` it frees the memory of each task from the (potentially big) `Alm` object.

# Arguments:
- `in_alm::Alm{T,Array{T,1}}`: `Alm` object to distribute over the MPI tasks.
- `out_d_alm::DAlm{T}`: output `DAlm` object.

# Keywords:
- `root::Integer`: rank of the task to be considered as "root", it is 0 by default.
- `clear::Bool`: if true deletes the input `Alm` after having performed the "scattering".
"""
function Scatter!(
    in_alm::Union{Healpix.Alm{T,Array{T,1}}, Vector{Healpix.Alm{T,Array{T,1}}}},
    out_d_alm::DAlm{S,T};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

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

function Scatter!(
    in_alm::Vector{Healpix.Alm{T,Array{T,1}}},
    out_d_alm::DAlm{S,T},
    out_d_pol_alm::DAlm{S,T};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    (length(in_alm) >= 3)||throw(DomainError(length(in_alm), "Input alm vector must have at least three components."))
    Scatter!(in_alm[1], out_d_alm, root = root, clear = clear)               #T
    Scatter!(view(in_alm, 2:3), out_d_pol_alm, root = root, clear = clear)   #E & B
end


#non root node
function Scatter!(
    in_alm::Nothing,
    out_d_alm::DAlm{S,T};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    comm = out_d_alm.info.comm
    (MPI.Comm_rank(comm) != root)||throw(DomainError(0, "Input alm on root task can not be `nothing`."))
    in_alm = MPI.bcast(nothing, root, comm)

    ScatterAlm!(in_alm, out_d_alm)

    if clear
        in_alm = nothing #free unnecessary copies of alm
    end
end

function Scatter!(
    ::Nothing,
    out_d_alm::DAlm{S,T},
    out_d_pol_alm::DAlm{S,T};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    Scatter!(nothing, out_d_alm, root = root, clear = clear)       #T
    Scatter!(nothing, out_d_pol_alm, root = root, clear = clear)   #E & B
end

function Scatter!(
    in_alm,
    out_d_alm::DAlm{S,T},
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}
    out_d_alm.info.comm = comm #overwrites the Comm in the d_alm
    MPI.Scatter!(in_alm, out_d_alm, root = root, clear = clear)
end

function Scatter!(
    in_alm,
    out_d_alm::DAlm{S,T},
    out_d_pol_alm::DAlm{S,T},
    comm::MPI.Comm;
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}
    out_d_alm.info.comm = comm #overwrites the Comm in the d_alm
    out_d_pol_alm.info.comm = comm
    MPI.Scatter!(in_alm, out_d_alm, out_d_pol_alm, root = root, clear = clear)
end
## GATHER
"""
    Internal function implementing a "Round Robin" strategy.

"""
function GatherAlm!(
    d_alm::DAlm{RR,T},
    alm::Healpix.Alm{T,Array{T,1}},
    root::Integer,
    comp::Integer
    ) where {T<:Number}

    (size(d_alm.alm, 2) >= comp) || throw(DomainError(size(d_alm.alm, 2), "not enough components in DAlm"))
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
            alm_chunk = @view d_alm.alm[i1:i2, comp] #@view speeds up the slicing
        else
            local_count = 0
            alm_chunk = ComplexF64[]
        end
        counts = MPI.Gather(Int32(local_count), root, comm) #FIXME: can this communication be avoided?
        outbuf = MPI.VBuffer(alm.alm, counts) #the output buffer points at the alms to overwrite
        outbuf.displs .+= displ_shift #we shift to the region in alm corresponding to the current round
        displ_shift += sum(counts) #we update the shift
        MPI.Gatherv!(alm_chunk, outbuf, root, comm) #gather the alms of this round
    end
end

function GatherAlm!(
    d_alm::DAlm{RR,T},
    ::Nothing,
    root::Integer,
    comp::Integer
    ) where {T<:Number}

    (size(d_alm.alm, 2) >= comp) || throw(DomainError(size(d_alm.alm, 2), "not enough components in DAlm"))
    (MPI.Comm_rank(d_alm.info.comm) != root)||throw(DomainError(0, "output alm on root task can not be `nothing`."))
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
            alm_chunk = @view d_alm.alm[i1:i2, comp] #@view speeds up the slicing
        else
            local_count = 0
            alm_chunk = ComplexF64[]
        end
        MPI.Gather(Int32(local_count), root, comm)
        MPI.Gatherv!(alm_chunk, nothing, root, comm) #gather the alms of this round
    end
end

"""
    Gather!(in_d_alm::DAlm{S,T}, out_alm::Union{Healpix.Alm{T,Array{T,1}}, Nothing}, comp::Integer; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Gather!(in_d_alm::DAlm{S,T}, out_alm::Healpix.Alm{T,Array{T,1}}; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Gather!(in_d_alm::DAlm{S,T}, out_alm::Vector{Healpix.Alm{T,Array{T,1}}}; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}
    Gather!(in_d_alm::DAlm{S,T}, out_alm::Nothing; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number}

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
function Gather!(
    in_d_alm::DAlm{S,T},
    out_alm::Union{Healpix.Alm{T,Array{T,1}}, Nothing},
    comp::Integer;
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    GatherAlm!(in_d_alm, out_alm, root, comp)
    if clear
        in_d_alm = nothing
    end
end

Gather!(in_d_alm::DAlm{S,T}, out_alm::Healpix.Alm{T,Array{T,1}}; root::Integer = 0, clear::Bool = false) where {S<:Strategy, T<:Number} =
    Gather!(in_d_alm, out_alm, 1, root = root, clear = clear)

#allows non-root tasks to pass nothing as output
function Gather!(
    in_d_alm::DAlm{S,T},
    out_alms::Vector{Healpix.Alm{T,Array{T,1}}};
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    size(in_d_alm.alm, 2) == length(out_alms)||throw(DomainError(length(out_alm), "Number of components in input and output alms not matching"))
    comp = 1
    for alm in out_alms
        GatherAlm!(in_d_alm, alm, root, comp)
        comp += 1
    end
    if clear
        in_d_alm = nothing
    end
end

function Gather!(
    in_d_alm::DAlm{S,T},
    ::Nothing;
    root::Integer = 0,
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    for comp in 1:size(in_d_alm.alm, 2)
        GatherAlm!(in_d_alm, nothing, root, comp)
    end
    if clear
        in_d_alm = nothing
    end
end

## Allgather
"""
    Internal function implementing a "Round Robin" strategy.
"""
function AllgatherAlm!(
    d_alm::DAlm{RR,T},
    alm::Healpix.Alm{T,Array{T,1}},
    comp::Integer
    ) where {T<:Number}

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
            alm_chunk = @view d_alm.alm[i1:i2, comp] #@view speeds up the slicing
        else
            local_count = 0
            alm_chunk = ComplexF64[]
        end
        counts = MPI.Allgather(Int32(local_count), comm)
        outbuf = MPI.VBuffer(alm.alm, counts) #the output buffer points at the alms to overwrite
        outbuf.displs .+= displ_shift #we shift to the region in alm corresponding to the current round
        displ_shift += sum(counts) #we update the shift

        MPI.Allgatherv!(alm_chunk, outbuf, comm) #gather the alms of this round
    end
end

"""
    Allgather!(in_d_alm::DAlm{S,T}, out_alm::Healpix.Alm{T,Array{T,1}}, comp::Integer; clear::Bool = false) where {S<:Strategy, T<:Number}
    Allgather!(in_d_alm::DAlm{S,T}, out_alm::Vector{Healpix.Alm{T,Array{T,1}}}, comp::Integer; clear::Bool = false) where {S<:Strategy, T<:Number}
    Allgather!(in_d_alm::DAlm{S,T}, out_alm::Healpix.Alm{T,Array{T,1}}; clear::Bool = false) where {S<:Strategy, T<:Number}

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
function Allgather!(
    in_d_alm::DAlm{S,T},
    out_alm::Healpix.Alm{T,Array{T,1}},
    comp::Integer;
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    AllgatherAlm!(in_d_alm, out_alm, comp)

    if clear
        in_d_alm = nothing
    end
end

Allgather!(in_d_alm::DAlm{S,T}, out_alm::Healpix.Alm{T,Array{T,1}}; clear::Bool = false) where {S<:Strategy, T<:Number} =
    Allgather!(in_d_alm, out_alm, 1, clear = clear)

function Allgather!(
    in_d_alm::DAlm{S,T},
    out_alms::Vector{Healpix.Alm{T,Array{T,1}}},
    comp::Integer;
    clear::Bool = false
    ) where {S<:Strategy, T<:Number}

    size(in_d_alm.alm, 2) == length(out_alms)||throw(DomainError(length(out_alm), "Number of components of input and output alms not matching"))
    comp = 1
    for out_alm in out_alms
        AllgatherAlm!(in_d_alm, out_alm, comp)
        comp += 1
    end
    if clear
        in_d_map = nothing
    end
end
#########################################################################
"""
	≃(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number}

Similarity operator, returns `true` if the two arguments have matching `info` objects.
"""
function ≃(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number}
    (&)((alm₁.info.comm == alm₂.info.comm),
        (alm₁.info.lmax == alm₂.info.lmax),
        (alm₁.info.mmax == alm₂.info.mmax),
        (alm₁.info.maxnm == alm₂.info.maxnm),
        (alm₁.info.mval == alm₂.info.mval),
        (alm₁.info.mstart == alm₂.info.mstart))
end

## DAlm Algebra

"""
    localdot(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number} -> Number

Internal function for the MPI-parallel dot product.
It performs a dot product LOCALLY on the current MPI task between the two
`DAlm`s passed in input.

"""
function localdot(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number}
    lmax = (alm₁.info.lmax == alm₂.info.lmax) ? alm₁.info.lmax : throw(DomainError(1, "lmax must match"))
    mval = (alm₁.info.mval == alm₂.info.mval) ? alm₁.info.mval : throw(DomainError(2, "mval must match"))
    mstart = (alm₁.info.mstart == alm₂.info.mstart) ? alm₁.info.mstart : throw(DomainError(3, "mstarts must match"))
    (size(alm₁.alm, 2) >= comp₁) || throw(DomainError(4, "not enough components in alm_1"))
    (size(alm₂.alm, 2) >= comp₂) || throw(DomainError(4, "not enough components in alm_2"))
    nm = length(mval)
    res_m0 = 0
    res_rest = 0
    @inbounds for mi in 1:nm #maybe run in parallel with Threads.@threads
        m = mval[mi]
        i1 = mstart[mi] + m
        i2 = i1 + lmax - m #this gives index range for each ell, for given m
        if (m == 0)
            @inbounds for i in i1:i2
                res_m0 += real(alm₁.alm[i, comp₁]) * real(alm₂.alm[i, comp₂]) #if m=0 we have no imag part
            end
        else
            @inbounds for i in i1:i2
                res_rest += real(alm₁.alm[i, comp₁]) * real(alm₂.alm[i, comp₂]) + imag(alm₁.alm[i, comp₁]) * imag(alm₂.alm[i, comp₂])
            end
        end
    end
    return res_m0 + 2*res_rest
end

import LinearAlgebra: dot
"""
    dot(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number} -> Number

MPI-parallel dot product between two `DAlm` object of matching size.
Use the `comp` keywords (defaulted to 1) to specify which component (column)
of each alm arrays is to be used for the computation.
"""
function dot(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number}
    comm = (alm₁.info.comm == alm₂.info.comm) ? alm₁.info.comm : throw(DomainError(0, "Communicators must match"))

    res = localdot(alm₁, alm₂, comp₁ = comp₁, comp₂ = comp₂)
    MPI.Allreduce(res, +, comm) #we sum together all the local results on each task
end

import Base: +, -, *, /

"""
    +(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number}

Perform the element-wise SUM of two `DAlm` objects in a_ℓm space.
A new `DAlm` object is returned.
"""
+(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(alm₁.alm .+ alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))

"""
    -(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number}

Perform the element-wise SUBTRACTION of two `DAlm` objects in a_ℓm space.
A new `DAlm` object is returned.
"""
-(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(alm₁.alm .- alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))

"""
    almxfl!(alm::DAlm{S,T}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, AA<:AbstractArray{N}}

Multiply IN-PLACE a subset of a_ℓm in the form of `DAlm` by a vector `fl`
representing an ℓ-dependent function.

# Arguments
- `alm::DAlm{S,T}`: The subset of spherical harmonics coefficients
- `fl`: The array giving the factor f_ℓ to multiply by a_ℓm, can be a `Vector{T}` or have as many columns as the components of `alm` we want to multiply

"""
function Healpix.almxfl!(alm::DAlm{S,T}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, AA<:AbstractArray{N}}

    lmax = alm.info.lmax
    mval = alm.info.mval
    fl_size = length(fl)

    if lmax + 1 > fl_size
        fl = [fl; zeros(lmax + 1 - fl_size)]
    end

    ncol = min(size(alm.alm, 2), size(fl, 2))
    Threads.@threads for col in 1:ncol
        i = 1
        @inbounds for m in mval
            for l = m:lmax
                alm.alm[i, col] = alm.alm[i, col]*fl[l+1, col]
                i += 1
            end
        end
    end
end

"""
    almxfl(alm::DAlm{S,T}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, AA<:AbstractArray{N,1}}

Multiply a subset of a_ℓm in the form of `DAlm` by a vector b_ℓ representing
an ℓ-dependent function, without changing the a_ℓm passed in input.

# Arguments
- `alm::DAlm{S,T}`: The array representing the spherical harmonics coefficients
- `fl::AbstractVector{T}`: The array giving the factor f_ℓ by which to multiply a_ℓm

# Returns
- `Alm{S,T}`: The result of a_ℓm * f_ℓ.
"""
function Healpix.almxfl(alm::DAlm{S,T}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, AA<:AbstractArray{N}}
    alm_new = deepcopy(alm)
    Healpix.almxfl!(alm_new, fl)
    alm_new
end

"""
    *(alm::DAlm{S,T}, fl::AA) where {S<:Strategy, T<:Number, AA<:AbstractArray{T,1}}
    *(fl::AA, alm::DAlm{S,T}) where {S<:Strategy, T<:Number, AA<:AbstractArray{T,1}}

Perform the MULTIPLICATION of a `DAlm` object by a function of ℓ in a_ℓm space.
Note: this consists in a shortcut of `almxfl`, therefore a new `DAlm`
object is returned.
"""
*(alm::DAlm{S,T}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, AA<:AbstractArray{N}} = Healpix.almxfl(alm, fl)
*(fl::AA, alm::DAlm{S,T}) where {S<:Strategy, T<:Number, N<:Number, AA<:AbstractArray{N}} = alm*fl

"""
    *(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number}
    *(alm₁::DAlm{S,T}, c::Number) where {S<:Strategy, T<:Number}
    *(c::Number, alm₁::DAlm{S,T}) where {S<:Strategy, T<:Number}

Perform the element-wise MULTIPLICATION of two `DAlm` objects or of a `DAlm` by a constant in a_ℓm space.
A new `DAlm` object is returned.
"""
*(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(alm₁.alm .* alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))
*(alm₁::DAlm{S,T}, c::Number) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(alm₁.alm .* c, alm₁.info)
*(c::Number, alm₁::DAlm{S,T}) where {S<:Strategy, T<:Number} = alm₁ * c

"""
    /(alm::DAlm{S,T}, fl::A1) where {S<:Strategy, T<:Number, N<:Number, A1<:AbstractArray{N,1}}
    /(alm::DAlm{S,T}, fl::A2) where {S<:Strategy, T<:Number, N<:Number, A2<:AbstractArray{N,2}}

Perform an element-wise DIVISION by a function of ℓ in a_ℓm space.
Note: this consists in a shortcut of `almxfl`, therefore a new `DAlm`
object is returned.
"""
/(alm::DAlm{S,T}, fl::AA) where {S<:Strategy, T<:Number, N<:Number, AA<:AbstractArray{N}} = Healpix.almxfl(alm, 1. ./ fl)

"""
    /(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number}
    /(alm₁::DAlm{S,T}, c::Number) where {S<:Strategy, T<:Number}

Perform the element-wise DIVISION of two `DAlm` objects or of a `DAlm` by a constant in a_ℓm space.
A new `DAlm` object is returned.
"""
/(alm₁::DAlm{S,T}, alm₂::DAlm{S,T}) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(alm₁.alm ./ alm₂.alm, alm₁ ≃ alm₂ ? alm₁.info : throw(DomainError(0,"info not matching")))
/(alm₁::DAlm{S,T}, c::Number) where {S<:Strategy, T<:Number} =
    DAlm{S,T}(alm₁.alm ./ c, alm₁.info)
