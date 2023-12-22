
"""
    Computes the local contribution to the power spectrum ``C_{\\ell}``.
"""
function alm2cl_local(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number, I<:Integer}
    (alm₁.info == alm₂.info) || throw(DomainError("infos not matching"))

    comp₁ = (size(alm₁.alm, 2) >= comp₁) ? comp₁ : throw(DomainError(4, "not enough components in alm_1"))
    comp₂ = (size(alm₂.alm, 2) >= comp₂) ? comp₂ : throw(DomainError(4, "not enough components in alm_2"))

    lmax = alm₁.info.lmax
    cl = zeros(Float64, lmax + 1)
    i = 1
    for m in alm₁.info.mval
        for l = m:lmax
            cl[l + 1] += ifelse(m > 0, 2, 1) * real(alm₁.alm[i, comp₁] * conj(alm₂.alm[i, comp₂])) / (2*l+1)
            i += 1
        end
    end
    cl
end

"""
    alm2cl(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number, I<:Integer} -> Vector{T}
    alm2cl(alm::DAlm{S,T,I}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number, I<:Integer} -> Vector{T}

Compute the power spectrum ``C_{\\ell}`` on each MPI task from the spherical harmonic
coefficients of one or two fields, distributed as `DAlm`.
Use the keywords `comp₁` and `comp₂` to specify which component (column) of the alms is
to be used for the computations

"""
function Healpix.alm2cl(alm₁::DAlm{S,T,I}, alm₂::DAlm{S,T,I}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number, I<:Integer}
    comm = (alm₁.info.comm == alm₂.info.comm) ? alm₁.info.comm : throw(DomainError(0, "Communicators must match"))

    local_cl = alm2cl_local(alm₁, alm₂, comp₁ = comp₁, comp₂ = comp₂)
    MPI.Allreduce(local_cl, +, comm)
end
Healpix.alm2cl(alm::DAlm{S,T,I}; comp₁::Integer = 1, comp₂::Integer = 1) where {S<:Strategy, T<:Number, I<:Integer} = Healpix.alm2cl(alm, alm, comp₁ = comp₁, comp₂ = comp₂)

import Random

"""
    synalm!(cl::Vector{T}, alm::DAlm{S,N,I}, rng::AbstractRNG; comp::Integer = 1) where {S<:Strategy, T<:Real, N<:Number, I<:Integer}
    synalm!(cl::Vector{T}, alm::DAlm{S,N,I}; comp::Integer = 1) where {S<:Strategy, T<:Real, N<:Number, I<:Integer}

Generate a set of `DAlm` from a given power spectra array `cl`.
The output is written into the `comp` column (defaulted to 1)
of the `Alm` object passed in input. If `comp` is greater than the number of
components (columns) in `Alm` a new column is appended to the alm matrix.
An RNG can be specified, otherwise it's defaulted to `Random.GLOBAL_RNG`.
"""
function Healpix.synalm!(cl::Vector{T}, alm::DAlm{S,N,I}, rng::Random.AbstractRNG; comp::Integer = 1) where {S<:Strategy, T<:Real, N<:Number, I<:Integer}
    cl_size = length(cl)
    lmax = alm.info.lmax
    mval = alm.info.mval
    (cl_size - 1 >= lmax) || throw(DomainError(cl_size, "not enough C_l's to generate Alm"))
    if comp > size(alm.alm, 2)      #if the desired comp is out of bounds
        cat(alm.alm, Array{T}(undef, length(alm.alm), 1), dims = 2)
        comp = size(alm.alm, 2) + 1 #we append a column next to the last one
    end
    i = 1
    for m in mval
        for l = m:lmax
            #for m=0 the alm must be real, since alm^R_l,0 = alm^C_l,0, if the field is real!
            alm.alm[i,comp] = Random.randn(rng, ifelse(m > 0, ComplexF64, Float64))*sqrt(cl[l+1]) #sqrt bc it's the variance
            i += 1
        end
    end
end
Healpix.synalm!(cl::Vector{T}, alm::DAlm{S,N,I}; comp::Integer = 1) where {S<:Strategy, T<:Real, N<:Number, I<:Integer} =
    Healpix.synalm!(cl, alm, Random.GLOBAL_RNG, comp=comp)
