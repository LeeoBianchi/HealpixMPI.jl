using Ducc0
import Healpix: alm2map!, adjoint_alm2map!

#round robin a2m communication
function communicate_alm2map!(in_leg::StridedArray{Complex{T},3}, out_leg::StridedArray{Complex{T},3}, comm::MPI.Comm, RR) where {T<:Real}
    c_size = MPI.Comm_size(comm)
    tot_nm, loc_nr, ncomp_out = size(out_leg) #global nm, local n rings
    loc_nm, tot_nr, ncomp_in = size(in_leg)  #local nm, global n rings,
    ncomp = (ncomp_in == ncomp_out) ? ncomp_in : throw(DomainError(0, "ncomp's of leg coefficinets must match"))
    eq_index = (tot_nr + 1) ÷ 2
    nside = eq_index ÷ 2
    tot_mmax = tot_nm-1

    #1) we compute the send and receive counts
    send_counts = Vector{Int64}(undef, c_size)
    rec_counts = Vector{Int64}(undef, c_size)
    rec_array = Vector{ComplexF64}(undef, tot_nm*loc_nr)
    cumrecs = Vector{Int64}(undef, c_size) #sort of cumulative sum corrected for the 1-base of the received counts, needed when unpacking
    cumrec = 1
    for t_rank in 0:c_size-1
        send_count = loc_nm * get_nrings_RR(eq_index, t_rank, c_size)
        send_counts[t_rank+1] = send_count
        rec_count = loc_nr * get_nm_RR(tot_mmax, t_rank, c_size) #local nrings x local nm on task t_rank
        rec_counts[t_rank+1] = rec_count
        cumrecs[t_rank+1] = cumrec
        cumrec += rec_count
    end
    
    for comp in 1:ncomp #cycle over the components of the leg (=1 if spin=0, =3 if spin=2 (TEB))
        #2) communicate
        #println("on task $(MPI.Comm_rank(comm)), we send $send_counts and receive $rec_counts coefficients")
        MPI.Alltoallv!(MPI.VBuffer(view(in_leg,:,:,comp), send_counts), MPI.VBuffer(rec_array, rec_counts), comm) #send_arr gets changed in place
        #3) unpack what we have received and fill out_leg
        ndone = zeros(Int, c_size) #keeps track of the processed elements coming from each task
        for ri in 1:loc_nr #local nrings
            @inbounds for mi in 1:tot_nm #tot nm
                send_idx = rem(mi-1, c_size) + 1 #1-based index of task who sent coefficients corresponding to that m
                idx = cumrecs[send_idx] + ndone[send_idx]
                ndone[send_idx] += 1
                out_leg[mi, ri, comp] = rec_array[idx]
            end
        end
    end
end

#for now we only support spin-0
"""
    alm2map!(d_alm::DAlm{S,N,I}, d_map::DMap{S,T,I}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    alm2map!(d_alm::DAlm{S,N,I}, d_map::DMap{S,T,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    alm2map!(d_alms::Vector{DAlm{S,N,I}}, d_pol_map::Vector{DMap{S,T,I}}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    alm2map!(d_alms::Vector{DAlm{S,N,I}}, d_pol_map::Vector{DMap{S,T,I}}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}

This function performs an MPI-parallel spherical harmonic transform, computing a distributed map from a set of `DAlm` and places the results
in the passed `d_map` object.

It must be called simultaneously on all the MPI tasks containing the subsets which form exactly the whole map and alm.

It is possible to pass two auxiliary arrays where the Legandre coefficients will be stored during the transform, this avoids allocating extra memory and improves efficiency.


# Arguments:

- `d_alm::DAlm{S,N,I}`: the MPI-distributed spherical harmonic coefficients to transform.

- `d_map::DMap{S,T,I}`: the MPI-distributed map that will contain the result.

# Optionals:

- `aux_in_leg::StridedArray{Complex{T},3}`: (local_nm, tot_nring, 1) auxiliary matrix for alm-side Legandre coefficients.

- `aux_out_leg::StridedArray{Complex{T},3}`: (tot_nm, local_nring, 1) auxiliary matrix for map-side Legandre coefficients.

# Keywords

- `nthreads::Integer = 0`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.
"""
function alm2map!(d_alm::DAlm{S,N,I}, d_map::DMap{S,T,I}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    comm = (d_alm.info.comm == d_map.info.comm) ? d_alm.info.comm : throw(DomainError(0, "Communicators must match"))
    #we first compute the leg's for local m's and all the rings (orderd fron N->S)
    in_alm = reshape(d_alm.alm, length(d_alm.alm), 1)
    cmval = Csize_t.(d_alm.info.mval)
    cmstart = Cptrdiff_t.(d_alm.info.mstart)
    lmax = d_alm.info.lmax
    thetatot = d_map.info.thetatot
    Ducc0.Sht.alm2leg!(in_alm, aux_in_leg, 0, lmax, cmval, cmstart, 1, thetatot, nthreads)
    #we transpose the leg's over tasks
    if MPI.Comm_size(comm) == 1
        aux_out_leg = aux_in_leg #we avoid useless communication on 1-task case
    else
        communicate_alm2map!(aux_in_leg, aux_out_leg, comm, S)
    end
    #then we use them to get the map
    Ducc0.Sht.leg2map!(aux_out_leg, reshape(d_map.pixels, :, 1), Csize_t.(d_map.info.nphi), d_map.info.phi0, Csize_t.(d_map.info.rstart), 1, nthreads)
end

function alm2map!(d_alm::DAlm{S,N,I}, d_map::DMap{S,T,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    aux_in_leg = Array{ComplexF64,3}(undef, (length(d_alm.info.mval), Healpix.numOfRings(d_map.info.nside), 1)) # loc_nm * tot_nr * 1
    aux_out_leg = Array{ComplexF64,3}(undef, d_alm.info.mmax+1, length(d_map.info.rings), 1)        # tot_nm * loc_nr * 1 #Check if Array{Array{T,2},1} could be faster
    Healpix.alm2map!(d_alm, d_map, aux_in_leg, aux_out_leg; nthreads = nthreads)
end

function alm2map!(d_alms::Vector{DAlm{S,N,I}}, d_pol_map::Vector{DMap{S,T,I}}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    #rifaccio tutto con solo una comunicazione? Chiamando poi alm2leg prima con spin=0 e poi spin=2
    #Faccio una volta sola la comunincazione
    #Come gestisco gli aux_leg?? Posso crearne solo 1 da dim = (r, m, 3) e poi passo 2 views a alm2leg? Temo di no... 
end

##################################################################################################
#MAP2ALM direction

#round robin adj communication
function communicate_map2alm!(in_leg::StridedArray{Complex{T},3}, out_leg::StridedArray{Complex{T},3}, comm::MPI.Comm, RR) where {T<:Real}
    c_size = MPI.Comm_size(comm)
    tot_nm, loc_nr, ncomp_in = size(in_leg) #global nm, local n rings
    loc_nm, tot_nr, ncomp_out = size(out_leg)   #local nm, global n rings
    ncomp = (ncomp_in == ncomp_out) ? ncomp_in : throw(DomainError(0, "ncomp's of leg coefficinets must match"))
    eq_index = (tot_nr + 1) ÷ 2
    tot_mmax = tot_nm-1

    #1) we pack the coefficients to send
    send_array = Vector{ComplexF64}(undef, tot_nm*loc_nr)
    send_counts = Vector{Int64}(undef, c_size)
    rec_counts = Vector{Int64}(undef, c_size)
    for comp in 1:ncomp  #cycle over the components of the leg (=1 if spin=0, =2 if spin=2)
        filled_leg = 1 #we keep track of how much of send_array we already have filled
        for t_rank in 0:c_size-1
            mindexes = get_mval_RR(tot_mmax, t_rank, c_size) .+ 1
            send_count = length(mindexes)*loc_nr
            send_matr = @view in_leg[mindexes, :, comp]  #chunk of leg to send to t_rank
            send_arr = @view send_array[filled_leg:filled_leg+send_count-1] #chunk of send_array to send to t_rank
            copyto!(send_arr, send_matr) #in-place version of reduce(vcat,...)
            send_counts[t_rank+1] = send_count
            rec_count = loc_nm * get_nrings_RR(eq_index, t_rank, c_size) #nrings on task t_rank
            rec_counts[t_rank+1] = rec_count
            filled_leg += send_count
        end
    #2) communicate
        #println("on task $(MPI.Comm_rank(comm)), we send $send_counts and receive $rec_counts coefficients")
        MPI.Alltoallv!(MPI.VBuffer(send_array, send_counts), MPI.VBuffer(view(out_leg,:,:,comp), rec_counts), comm)
    end
end

"""
    adjoint_alm2map!(d_map::DMap{S,T,I}, d_alm::DAlm{S,N,I}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    adjoint_alm2map!(d_map::DMap{S,T,I}, d_alm::DAlm{S,N,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    adjoint_alm2map!(d_pol_map::Vector{DMap{S,T,I}}, d_alms::Vector{DAlm{S,N,I}}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}

This function performs an MPI-parallel spherical harmonic transform Yᵀ on the distributed map and places the results
in the passed `d_alm` object.

It must be called simultaneously on all the MPI tasks containing the subsets which form exactly the whole map and alm.

It is possible to pass two auxiliary arrays where the Legandre coefficients will be stored during the transform, this avoids allocating extra memory and improves efficiency.

# Arguments:

- `d_map::DMap{S,T,I}`: the distributed map that must be decomposed in spherical harmonics.

- `alm::Alm{ComplexF64, Array{ComplexF64, 1}}`: the spherical harmonic
  coefficients to be written to.

# Optionals:

- `aux_in_leg::StridedArray{Complex{T},3}`: (local_nm, tot_nring, 1) auxiliary matrix for map-side Legandre coefficients.

- `aux_out_leg::StridedArray{Complex{T},3}`: (tot_nm, local_nring, 1) auxiliary matrix for alm-side Legandre coefficients.

# Keywords

- `nthreads::Integer = 0`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.
"""
function adjoint_alm2map!(d_map::DMap{S,T,I}, d_alm::DAlm{S,N,I}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    comm = (d_alm.info.comm == d_map.info.comm) ? d_alm.info.comm : throw(DomainError(0, "Communicators must match"))
    #compute leg
    Ducc0.Sht.map2leg!(reshape(d_map.pixels, length(d_map.pixels), 1), aux_in_leg, Csize_t.(d_map.info.nphi), d_map.info.phi0, Csize_t.(d_map.info.rstart), 1, nthreads)
    #we transpose the leg's over tasks
    if MPI.Comm_size(comm) == 1
        aux_out_leg = aux_in_leg #we avoid useless communication on 1-task case
    else
        communicate_map2alm!(aux_in_leg, aux_out_leg, comm, S)
    end
    #then we use them to get the alm
    Ducc0.Sht.leg2alm!(aux_out_leg, reshape(d_alm.alm, :, 1), 0, d_alm.info.lmax, Csize_t.(d_alm.info.mval), Cptrdiff_t.(d_alm.info.mstart), 1, d_map.info.thetatot, nthreads)
end

function adjoint_alm2map!(d_map::DMap{S,T,I}, d_alm::DAlm{S,N,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    aux_in_leg = Array{ComplexF64,3}(undef, d_alm.info.mmax+1, length(d_map.info.rings), 1)               # tot_nm * loc_nr
    aux_out_leg = Array{ComplexF64,3}(undef, (length(d_alm.info.mval), Healpix.numOfRings(d_map.info.nside), 1))  # loc_nm * tot_nr
    Healpix.adjoint_alm2map!(d_map, d_alm, aux_in_leg, aux_out_leg; nthreads = nthreads)
end

function adjoint_alm2map!(d_pol_map::Vector{DMap{S,T,I}}, d_alms::Vector{DAlm{S,N,I}}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}

end
