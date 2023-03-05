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
else
    test_map = nothing
end

d_map = DMap{RR}()
MPI.Scatter!(test_map, d_map, comm)

#TEST ALGEBRA
d_map2 = 2. *d_map

Test.@test (d_map + d_map).pixels == d_map2.pixels
Test.@test (d_map2/2.).pixels == (d_map2 - d_map).pixels
Test.@test isapprox((d_map * d_map / d_map).pixels, d_map.pixels)

Test.@test isapprox(((π + d_map) - π).pixels, d_map.pixels)
