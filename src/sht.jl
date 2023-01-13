include("/home/leoab/OneDrive/UNI/ducc/julia/ducc0.jl") #FIXME: replace with proper binding to Ducc0

include("alm.jl")
include("map.jl")

import Healpix: alm2map! #adjoint_alm2map!, when it will be added in Healpix.jl

function communicate_alm2map!(in_leg::StridedArray{Complex{T},3}, out_leg::StridedArray{Complex{T},3}, comm::MPI.Comm) where {T<:Real}
    c_size = MPI.Comm_size(comm)
    loc_nm = size(in_leg, 1)  #local nm
    tot_nm = size(out_leg, 1) #global nm
    loc_nr = size(out_leg, 2) #local n rings
    tot_nr = size(in_leg, 2)  #global n rings
    eq_index = (tot_nr + 1) ÷ 2
    tot_mmax = tot_nm-1

    #1) we pack the coefficients to send
    send_array = Vector{ComplexF64}(undef, size(in_leg, 1)*size(in_leg, 2))
    send_counts = Vector{Int64}(undef, c_size)
    rec_counts = Vector{Int64}(undef, c_size)
    cumrecs = Vector{Int64}(undef, c_size) #sort of cumulative sum corrected for the 1-base of the received counts, needed when unpacking
    filled = 1 #we keep track of how much of send_array we already have filled
    cumrec = 1
    for t_rank in 0:c_size-1
        local_chunk = reduce(vcat, in_leg[:, get_rindexes_RR(loc_nr, eq_index, t_rank, c_size), 1])
        send_count = length(local_chunk)
        send_array[filled:filled+send_count-1] = local_chunk
        send_counts[t_rank+1] = send_count
        rec_count = loc_nr * get_nm_RR(tot_mmax, t_rank, c_size) #local nrings x local nm on task t_rank
        rec_counts[t_rank+1] = rec_count
        cumrecs[t_rank+1] = cumrec
        cumrec += rec_count
        filled += send_count
    end
    #2) communicate
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
function alm2map!(d_alm::DistributedAlm{Complex{T},I}, d_map::DistributedMap{T,I}; nthreads = 0) where {T<:Real, I<:Integer}
    comm = (d_alm.info.comm == d_map.info.comm) ? d_alm.info.comm : throw(DomainError(0, "Communicators must match"))
    theta = d_map.info.thetatot #global list of theta, ordered N->S

    #we first compute the leg's for local m's and all the rings (orderd fron N->S)
    #FIXME: maybe implement fast type conversion inside Ducc0.sht_alm2leg
    in_leg = Ducc0.sht_alm2leg(reshape(d_alm.alm, length(d_alm.alm), 1), Unsigned(0), Unsigned(d_alm.info.lmax), Csize_t.(d_alm.info.mval), Csize_t.(d_alm.info.mstart), 1, d_map.info.theta, Unsigned(nthreads))

    #we transpose the leg's over tasks
    #FIXME: maybe add leg as a field of Distributed* classes, so we avoid creation every time
    out_leg = Array{ComplexF64,3}(undef, d_alm.info.mmax+1, length(d_map.info.rings), 1) # tot_nm * loc_nr Matrix.
    communicate_alm2map!(in_leg, out_leg, comm)

    #then we use them to get the map
    d_map.pixels = Ducc0.sht_leg2map(out_leg, Csize_t.(d_map.info.nphi), d_map.info.phi0, Csize_t.(d_map.info.rstart), 1, Unsigned(nthreads))[:,1]
end

function communicate_map2alm(args)
    body
end

function adjoint_alm2map!(args)
    body #NOTE: remember that our rstart is 1-based and Ducc wants it 0-based
end
