using Healpix
using MPI
using Test

include("../src/map.jl")

#################################################################

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    test_map = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(2)])
    res_map = HealpixMap{Float64, RingOrder}(zeros(nside2npix(2)))
else
    test_map = nothing
    res_map = nothing
end

test_map_all = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(2)])
res_map_all = HealpixMap{Float64, RingOrder}(zeros(nside2npix(2)))

d_map = DistributedMap()
MPI.Scatter!(test_map, d_map, comm)

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

#TEST ALGEBRA
d_map2 = 2. *d_map

Test.@test (d_map + d_map).pixels == d_map2.pixels
Test.@test (d_map2/2.).pixels == (d_map2 - d_map).pixels
Test.@test isapprox((d_map * d_map / d_map).pixels, d_map.pixels)

Test.@test isapprox(((π + d_map) - π).pixels, d_map.pixels)
