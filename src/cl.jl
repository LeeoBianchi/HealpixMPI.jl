using MPI
using Random
include("alm.jl")

"""
    Computes the local contribution to the power spectrum ``C_{\\ell}``.
"""
function alm2cl_local(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    (alm₁.info == alm₂.info) || throw(DomainError("infos not matching"))

    lmax = alm₁.info.lmax
    cl = zeros(T, lmax + 1)
    i = 1
    for m in alm₁.info.mval
        for l = m:lmax
            cl[l + 1] += ifelse(m > 0, 2, 1) * real(alm₁.alm[i] * conj(alm₂.alm[i])) / (2*l+1)
            i += 1
        end
    end
    cl
end

import Healpix: alm2cl
"""
    alm2cl(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer} -> Vector{T}
    alm2cl(alm::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer} -> Vector{T}

Compute the power spectrum ``C_{\\ell}`` on each MPI task from the spherical harmonic
coefficients of one or two fields, distributed as `DistributedAlm`.

"""
function alm2cl(alm₁::DistributedAlm{Complex{T},I}, alm₂::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer}
    comm = (alm₁.info.comm == alm₂.info.comm) ? alm₁.info.comm : throw(DomainError(0, "Communicators must match"))

    local_cl = alm2cl_local(alm₁, alm₂)
    MPI.Barrier(comm) #FIXME: necessary??
    MPI.Allreduce(local_cl, +, comm)
end
alm2cl(alm::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer} = alm2cl(alm, alm)


"""
    synalm!(cl::Vector{T}, alm::DistributedAlm{Complex{T},I}, rng::AbstractRNG) where {T<:Real, I<:Integer}
    synalm!(cl::Vector{T}, alm::DistributedAlm{Complex{T},I}) where {T <: Real}

Generate a set of `DistributedAlm` from a given power spectra array `cl`.
The output is written into the `Alm` object passed in input.
An RNG can be specified, otherwise it's defaulted to `Random.GLOBAL_RNG`.
"""
function synalm!(cl::Vector{T}, alm::DistributedAlm{Complex{T},I}, rng::AbstractRNG) where {T<:Real, I<:Integer}
    cl_size = length(cl)
    lmax = alm.info.lmax
    mval = alm.info.mval
    (cl_size - 1 >= lmax) || throw(DomainError(cl_size, "not enough C_l's to generate Alm"))

    i = 1
    for m in mval
        for l = m:lmax
            #for m=0 the alm must be real, since alm^R_l,0 = alm^C_l,0, if the field is real!
            alm.alm[i] = randn(rng, ifelse(m > 0, ComplexF64, Float64))*sqrt(cl[l+1]) #sqrt bc it's the variance
            i += 1
        end
    end
end
synalm!(cl::Vector{T}, alm::DistributedAlm{Complex{T},I}) where {T<:Real, I<:Integer} =
    synalm!(cl, alm, Random.GLOBAL_RNG)
