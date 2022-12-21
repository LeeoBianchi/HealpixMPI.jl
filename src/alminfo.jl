### overloading getproperty:
getlmax(alminfo::Cptr) = alminfo.lmax[]
getlmax(alminfo::AlmInfo) = getlmax(reinterpret(Cptr{sharp_alm_info}, alminfo.ptr))

getnm(alminfo::Cptr) = alminfo.nm[]
getnm(alminfo::AlmInfo) = getnm(reinterpret(Cptr{sharp_alm_info}, alminfo.ptr)) #NOTE: questo sembra il metodo vincente!

getmval(alminfo::Cptr{sharp_alm_info}) = alminfo.mval[]
getmval(alminfo::Ptr{Nothing}) = getmval(reinterpret(Cptr{sharp_alm_info}, alminfo))
getmval(alminfo::AlmInfo) = unsafe_wrap(Array, reinterpret(Ptr{Cint}, getmval(alminfo.ptr)), getnm(alminfo))

getflags(alminfo::Cptr) = alminfo.flags[]
getflags(alminfo::AlmInfo) = getflags(reinterpret(Cptr{sharp_alm_info}, alminfo.ptr))

getmvstart(alminfo::Cptr{sharp_alm_info}) = alminfo.mvstart[]
getmvstart(alminfo::Ptr{Nothing}) = getmvstart(reinterpret(Cptr{sharp_alm_info}, alminfo))
getmvstart(alminfo::AlmInfo) = unsafe_wrap(Array, reinterpret(Ptr{Cptrdiff_t}, getmvstart(alminfo.ptr)), getnm(alminfo))

getstride(alminfo::Cptr) = alminfo.stride[]
getstride(alminfo::AlmInfo) = getstride(reinterpret(Cptr{sharp_alm_info}, alminfo.ptr))

function Base.getproperty(info::AlmInfo, name::Symbol)
    if     name === :lmax
        return getlmax(info)
    elseif name === :nm
        return getnm(info)
    elseif name === :mval
        return getmval(info)
    elseif name === :flags
        return getflags(info)
    elseif name === :mvstart
        return getmvstart(info)
    elseif name === :stride
        return getstride(info)
    else
        return getfield(info, name)
    end
end

### property names
function Base.propertynames(info::AlmInfo)
  return (:ptr, :lmax, :nm, :mval, :flags, :mvstart, :stride)
end

### show
function Base.show(io::IO, info::AlmInfo)
  print(io, "AlmInfo:
lmax = $(info.lmax)
nm = $(info.nm)
mval = $(info.mval)
flags = $(info.flags)
mvstart = $(info.mvstart)
stride = $(info.stride)")
end

#
""" make_mvstart_complex(lmax::Integer, stride::Integer, mval::AbstractArray{T}) where T <: Integer

    Computes the `mstart` array given any `mval` and `lmax` for `Alm` in complex
    representation.
"""
function make_mvstart_complex(lmax::Integer, stride::Integer, mval::AbstractArray{T}) where T <: Integer
    idx = 0 #tracks the number of 'consumed' elements so far; need to correct by m
    mi = 1 #index of mstart array
    mstart = Vector{Int}(undef, length(mval))
    for m in mval
        mstart[mi] = stride * (idx - m) #fill mstart
        idx += lmax + 1 - m
        mi += 1
    end
    mstart
end

""" make_mmajor_complex_alm_info(lmax::Integer, stride::Integer, mval::AbstractArray{T}) where T <: Integer

    Creates an `AlmInfo` object for a (sub)set of a_ℓm stored as complex numbers
    ordered by m-major (as in Healpix.jl), for any given array of m values.
    If mval is not passed it will be defaulted to the full range [0:lmax]

    # Arguments
    - `lmax::Integer`: maximum spherical harmonic ℓ
    - `stride::Integer`: the stride between consecutive ℓ's
    - `mval::AbstractArray{T}`: array containing the values of m included in the (sub)set
        pass `nothing` to use all the m's from 0 to lmax.

    # Returns
    - `AlmInfo` object
"""
function make_mmajor_complex_alm_info(
    lmax::Integer,
    stride::Integer,
    mval::AbstractArray{T}
    ) where T <: Integer

    nm = length(mval)
    mstart = make_mvstart_complex(lmax, stride, mval)
    make_general_alm_info(lmax, nm, stride, mval, mstart) #construct AlmInfo
end

make_mmajor_complex_alm_info(lmax::Integer, stride::Integer) =
    make_mmajor_complex_alm_info(lmax, stride, 0:lmax)

    """
        make_general_alm_info(
            lmax::Integer, nm::Integer, stride::Integer, mval::AbstractArray{T}, mstart::AbstractArray{T}
            ) where T <: Integer

    Initialises a general a_lm data structure according to the following parameter.
    It can be used to construct an `AlmInfo` object for a subset of an `Alm` set.

    # Arguments
    - `lmax::Integer`: maximum spherical harmonic ℓ
    - `nm::Integer`: number of different m values
    - `stride::Integer`: the stride between consecutive ℓ's
    - `mval::AbstractArray{T}`: array with `nm` entries containing the individual m values
    - `mstart::AbstractArray{T}`: array with `nm` entries containing the (hypothetical)
        indices {i} of the coefficients with the quantum numbers ℓ=0, m=mval[i]

    # Returns
    - `AlmInfo` object
    """
    function make_general_alm_info(
        lmax::Integer,
        nm::Integer,
        stride::Integer, #generally = 1
        mval::AbstractArray{T},
        mstart::AbstractArray{T}
        ) where T <: Integer

        alm_info_ptr = Ref{Ptr{Cvoid}}()
        mval_cint = [Cint(x) for x in mval]
        mstart_cptrdiff = [Cptrdiff_t(x) for x in mstart]

        ccall(
            (:sharp_make_general_alm_info, libsharp2),
            Cvoid,
            (Cint, Cint, Cint, Ref{Cint}, Ref{Cptrdiff_t}, Cint, Ref{Ptr{Cvoid}}),
            lmax, nm, stride, mval_cint, mstart_cptrdiff, 0, alm_info_ptr,
        )

        AlmInfo(alm_info_ptr[])
    end
