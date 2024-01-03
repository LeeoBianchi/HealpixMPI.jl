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
NSIDE = 8

if crank == root
    test_map = PolarizedHealpixMap{Float64, RingOrder}(
    [Float64(i) for i in 1:nside2npix(NSIDE)],
    [Float64(i) for i in 1:nside2npix(NSIDE)],
    [Float64(i) for i in 1:nside2npix(NSIDE)])
    res_map = PolarizedHealpixMap{Float64, RingOrder}(NSIDE)
else
    test_map = nothing
    res_map = nothing
end

test_map_all = PolarizedHealpixMap{Float64, RingOrder}(
    [Float64(i) for i in 1:nside2npix(NSIDE)],
    [Float64(i) for i in 1:nside2npix(NSIDE)],
    [Float64(i) for i in 1:nside2npix(NSIDE)])
res_map_all = PolarizedHealpixMap{Float64, RingOrder}(NSIDE)

d_map = DMap{RR}(comm) #inizialize empty DMap
d_map_pol = DMap{RR}(comm)

MPI.Scatter!(test_map, d_map, d_map_pol, comm) #fill it

MPI.Gather!(d_map, d_map_pol, res_map, clear = true) #re-gather
MPI.Allgather!(d_map, d_map_pol, res_map_all, clear = false)

MPI.Barrier(comm)

#TEST SCATTER/GATHER
if crank == root
    Test.@test test_map.i == res_map.i
    Test.@test test_map.q == res_map.q
    Test.@test test_map.u == res_map.u
else
    Test.@test res_map === nothing
end
Test.@test res_map_all.i == test_map_all.i
Test.@test res_map_all.q == test_map_all.q
Test.@test res_map_all.u == test_map_all.u
