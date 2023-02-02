using Test
using MPI
using Healpix
include("../src/HealpixMPI.jl")
using Main.HealpixMPI

#################################################################

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0
nside = 8

if crank == root
    test_map = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(nside)])
    res_map = HealpixMap{Float64, RingOrder}(zeros(nside2npix(nside)))
else
    test_map = nothing
    res_map = nothing
end

test_map_all = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(nside)])
res_map_all = HealpixMap{Float64, RingOrder}(zeros(nside2npix(nside)))

d_map = DistributedMap{RR}() #inizialize empty DistributedMap
MPI.Scatter!(test_map, d_map, comm) #fill it

MPI.Gather!(d_map, res_map, clear=false)

MPI.Allgather!(d_map, res_map_all, clear=false)

MPI.Barrier(comm)

#TEST SCATTER/GATHER
if crank == root
    Test.@test test_map == res_map
else
    Test.@test res_map === nothing
end
Test.@test res_map_all == test_map_all #FIXME: check why it doesn't work on a single taks.
