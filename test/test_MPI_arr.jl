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

#we test that the scattered array elements match the corresponding DMap pixels
nside = 8

if crank == root
    test_arr = Vector{Float64}([i for i in 1:nside2npix(nside)])
    test_map = HealpixMap{Float64, RingOrder}(test_arr)
else
    test_arr = nothing
    test_map = nothing
end

d_map = DMap{RR}()
MPI.Scatter!(test_map, d_map, comm)
local_arr = MPI.Scatter(test_arr, nside, comm)

Test.@test local_arr == d_map.pixels
