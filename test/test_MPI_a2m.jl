using Test
using MPI
using Healpix
include("../src/HealpixMPI.jl")
using Main.HealpixMPI

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    test_map = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(4)])
    test_alm = Alm(5, 5, [ComplexF64(i) for i in 1:numberOfAlms(5)])
else
    test_map = nothing
    test_alm = nothing
end

d_map = DistributedMap{RR}(comm)
MPI.Scatter!(test_map, d_map, comm)
d_alm = DistributedAlm{RR}(comm)
MPI.Scatter!(test_alm, d_alm, comm)

MPI.Barrier(comm)

# TEST ALM2MAP DIRECTION

alm2map!(d_alm, d_map; nthreads = 1)
out_map = deepcopy(test_map)
MPI.Gather!(d_map, out_map)

if crank == root
    Test.@test isapprox(out_map.pixels, alm2map(test_alm, 4).pixels) #test against Healpix transform
end
