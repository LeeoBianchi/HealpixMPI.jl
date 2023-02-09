using Ducc0

#import Healpix: alm2map! #adjoint_alm2map!, when it will be added in Healpix.jl

#round robin a2m communication
function communicate_alm2map!(in_leg::StridedArray{Complex{T},3}, out_leg::StridedArray{Complex{T},3}, comm::MPI.Comm, RR) where {T<:Real}
    c_size = MPI.Comm_size(comm)
    tot_nm, loc_nr = size(out_leg) #global nm, local n rings
    loc_nm, tot_nr = size(in_leg)  #local nm, global n rings,
    eq_index = (tot_nr + 1) ÷ 2
    nside = eq_index ÷ 2
    tot_mmax = tot_nm-1

    #1) we pack the coefficients to send
    send_array = Vector{ComplexF64}(undef, loc_nm*tot_nr)
    send_counts = Vector{Int64}(undef, c_size)
    rec_counts = Vector{Int64}(undef, c_size)
    cumrecs = Vector{Int64}(undef, c_size) #sort of cumulative sum corrected for the 1-base of the received counts, needed when unpacking
    filled = 1 #we keep track of how much of send_array we already have filled
    cumrec = 1
    for t_rank in 0:c_size-1
        rindexes = get_rindexes_RR(nside, t_rank, c_size)
        send_count = length(rindexes)*loc_nm
        send_matr = @view in_leg[:,rindexes,1]
        send_arr = @view send_array[filled:filled+send_count-1]
        copyto!(send_arr, send_matr) #in-place version of reduce(vcat,...)
        send_counts[t_rank+1] = send_count
        rec_count = loc_nr * get_nm_RR(tot_mmax, t_rank, c_size) #local nrings x local nm on task t_rank
        rec_counts[t_rank+1] = rec_count
        cumrecs[t_rank+1] = cumrec
        cumrec += rec_count
        filled += send_count
    end
    #2) communicate
    #println("on task $(MPI.Comm_rank(comm)), we send $send_counts and receive $rec_counts coefficients")
    received_array = MPI.Alltoallv(send_array, send_counts, rec_counts, comm)

    #3) unpack what we have received and fill out_leg
    ndone = zeros(Int, c_size) #keeps track of the processed elements coming from each task
    for ri in 1:loc_nr #local nrings
        for mi in 1:tot_nm #tot nm
            send_idx = rem(mi-1, c_size) + 1 #1-based index of task who sent coefficients corresponding to that m
            idx = cumrecs[send_idx] + ndone[send_idx]
            ndone[send_idx] += 1
            out_leg[mi, ri, 1] = received_array[idx]
        end
    end
end
#for now we only support spin-0
"""
    alm2map!(d_alm::DistributedAlm{S,N,I}, d_map::DistributedMap{S,T,I}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    alm2map!(d_alm::DistributedAlm{S,N,I}, d_map::DistributedMap{S,T,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}

This function performs an MPI-parallel spherical harmonic transform, computing a distributed map from a set of `DistributedAlm` and places the results
in the passed `d_map` object.

It must be called simultaneously on all the MPI tasks containing the subsets which form exactly the whole map and alm.

It is possible to pass two auxiliary arrays where the Legandre coefficients will be stored during the transform, this avoids allocating extra memory and improves efficiency.


# Arguments:

- `d_alm::DistributedAlm{S,N,I}`: the MPI-distributed spherical harmonic coefficients to transform.

- `d_map::DistributedMap{S,T,I}`: the MPI-distributed map that will contain the result.

# Optionals:

- `aux_in_leg::StridedArray{Complex{T},3}`: (local_nm, tot_nring, 1) auxiliary matrix for alm-side Legandre coefficients.

- `aux_out_leg::StridedArray{Complex{T},3}`: (tot_nm, local_nring, 1) auxiliary matrix for map-side Legandre coefficients.

# Keywords

- `nthreads::Integer = 0`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.
"""
function Healpix.alm2map!(d_alm::DistributedAlm{S,N,I}, d_map::DistributedMap{S,T,I}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    comm = (d_alm.info.comm == d_map.info.comm) ? d_alm.info.comm : throw(DomainError(0, "Communicators must match"))
    #we first compute the leg's for local m's and all the rings (orderd fron N->S)
    in_alm=reshape(d_alm.alm, length(d_alm.alm), 1)
    Ducc0.Sht.alm2leg!(in_alm, aux_in_leg, 0, d_alm.info.lmax, Csize_t.(d_alm.info.mval), Cptrdiff_t.(d_alm.info.mstart), 1, d_map.info.thetatot, nthreads)
    #we transpose the leg's over tasks
    MPI.Barrier(comm)
    #println("on task $(MPI.Comm_rank(comm)), we have in_leg with shape $(size(aux_in_leg)) and out_leg $(size(aux_out_leg))")
    communicate_alm2map!(aux_in_leg, aux_out_leg, comm, S)
    #then we use them to get the map
    Ducc0.Sht.leg2map!(aux_out_leg, reshape(d_map.pixels, :, 1), Csize_t.(d_map.info.nphi), d_map.info.phi0, Csize_t.(d_map.info.rstart), 1, nthreads)
end

function Healpix.alm2map!(d_alm::DistributedAlm{S,N,I}, d_map::DistributedMap{S,T,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    aux_in_leg = Array{ComplexF64,3}(undef, (length(d_alm.info.mval), Healpix.numOfRings(d_map.info.nside), 1)) # loc_nm * tot_nr
    aux_out_leg = Array{ComplexF64,3}(undef, d_alm.info.mmax+1, length(d_map.info.rings), 1)            # tot_nm * loc_nr
    Healpix.alm2map!(d_alm, d_map, aux_in_leg, aux_out_leg; nthreads = nthreads)
end

##################################################################################################
#MAP2ALM direction

#round robin adj communication
function communicate_map2alm!(in_leg::StridedArray{Complex{T},3}, out_leg::StridedArray{Complex{T},3}, comm::MPI.Comm, RR) where {T<:Real}
    c_size = MPI.Comm_size(comm)
    tot_nm, loc_nr = size(in_leg) #global nm, local n rings
    loc_nm, tot_nr = size(out_leg)   #local nm, global n rings
    eq_index = (tot_nr + 1) ÷ 2
    tot_mmax = tot_nm-1

    #1) we pack the coefficients to send
    send_array = Vector{ComplexF64}(undef, size(in_leg, 1)*size(in_leg, 2))
    rings_received = Vector{Int64}(undef, tot_nr) #Array storing the ring indexes in the order in which the rings are received, maybe implement similar for m's in communicate_alm2map
    send_counts = Vector{Int64}(undef, c_size)
    rec_counts = Vector{Int64}(undef, c_size)
    filled_leg = 1 #we keep track of how much of send_array we already have filled
    filled_ring = 1 #we keep track of how much of rings_received we already have filled
    for t_rank in 0:c_size-1
        mindexes = get_mval_RR(tot_mmax, t_rank, c_size) .+ 1
        send_count = length(mindexes)*loc_nr
        send_matr = @view in_leg[mindexes, :, 1]  #chunk of leg to send to t_rank
        send_arr = @view send_array[filled_leg:filled_leg+send_count-1] #chunk of send_array to send to t_rank
        copyto!(send_arr, send_matr) #in-place version of reduce(vcat,...)
        send_counts[t_rank+1] = send_count
        nring_t = get_nrings_RR(eq_index, t_rank, c_size) #nrings on task t_rank
        rec_count = loc_nm * nring_t
        rec_counts[t_rank+1] = rec_count
        rings_received[filled_ring:filled_ring+nring_t-1] = get_rindexes_RR(nring_t, eq_index, t_rank, c_size) #ring indexes sent from task t_rank
        filled_leg += send_count
        filled_ring += nring_t
    end

    #2) communicate
    #println("on task $(MPI.Comm_rank(comm)), we send $send_counts and receive $rec_counts coefficients")
    received_array = MPI.Alltoallv(send_array, send_counts, rec_counts, comm)

    #3) unpack what we have received and fill out_leg
    out_leg[:,:,1] = reshape(received_array, loc_nm, :)
    rings_received
end

"""
    adjoint_alm2map!(d_map::DistributedMap{S,T,I}, d_alm::DistributedAlm{S,N,I}; nthreads = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    adjoint_alm2map!(d_map::DistributedMap{S,T,I}, d_alm::DistributedAlm{S,N,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}

This function performs an MPI-parallel spherical harmonic transform Yᵀ on the distributed map and places the results
in the passed `d_alm` object.

It must be called simultaneously on all the MPI tasks containing the subsets which form exactly the whole map and alm.

It is possible to pass two auxiliary arrays where the Legandre coefficients will be stored during the transform, this avoids allocating extra memory and improves efficiency.

# Arguments:

- `d_map::DistributedMap{S,T,I}`: the distributed map that must be decomposed in spherical harmonics.

- `alm::Alm{ComplexF64, Array{ComplexF64, 1}}`: the spherical harmonic
  coefficients to be written to.

# Optionals:

- `aux_in_leg::StridedArray{Complex{T},3}`: (local_nm, tot_nring, 1) auxiliary matrix for map-side Legandre coefficients.

- `aux_out_leg::StridedArray{Complex{T},3}`: (tot_nm, local_nring, 1) auxiliary matrix for alm-side Legandre coefficients.

# Keywords

- `nthreads::Integer = 0`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.
"""
function Healpix.adjoint_alm2map!(d_map::DistributedMap{S,T,I}, d_alm::DistributedAlm{S,N,I}, aux_in_leg::StridedArray{Complex{T},3}, aux_out_leg::StridedArray{Complex{T},3}; nthreads = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    comm = (d_alm.info.comm == d_map.info.comm) ? d_alm.info.comm : throw(DomainError(0, "Communicators must match"))
    #compute leg
    Ducc0.Sht.map2leg!(reshape(d_map.pixels, length(d_map.pixels), 1), aux_in_leg, Csize_t.(d_map.info.nphi), d_map.info.phi0, Csize_t.(d_map.info.rstart), 1, nthreads)
    #we transpose the leg's over tasks
    #MPI.Barrier(comm)
    #println("on task $(MPI.Comm_rank(comm)), we have in_leg with shape $(size(aux_in_leg)) and out_leg $(size(aux_out_leg))")
    rings_received = communicate_map2alm!(aux_in_leg, aux_out_leg, comm, S) #additional output for reordering thetatot
    theta_reordered = d_map.info.thetatot[rings_received] #colatitudes ordered by task first and RR within each task
    #then we use them to get the alm
    Ducc0.Sht.leg2alm!(aux_out_leg, reshape(d_alm.alm, :, 1), 0, d_alm.info.lmax, Csize_t.(d_alm.info.mval), Cptrdiff_t.(d_alm.info.mstart), 1, theta_reordered, nthreads)
end

function Healpix.adjoint_alm2map!(d_map::DistributedMap{S,T,I}, d_alm::DistributedAlm{S,N,I}; nthreads::Integer = 0) where {S<:Strategy, N<:Number, T<:Real, I<:Integer}
    aux_in_leg = Array{ComplexF64,3}(undef, d_alm.info.mmax+1, length(d_map.info.rings), 1)               # tot_nm * loc_nr
    aux_out_leg = Array{ComplexF64,3}(undef, (length(d_alm.info.mval), Healpix.numOfRings(d_map.info.nside), 1))  # loc_nm * tot_nr
    Healpix.adjoint_alm2map!(d_map, d_alm, aux_in_leg, aux_out_leg; nthreads = nthreads)
end
