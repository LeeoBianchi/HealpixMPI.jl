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

NSIDE = 64
lmax = 3*NSIDE - 1

if crank == root
    test_map = HealpixMap{Float64, RingOrder}(NSIDE)
    test_alm = Alm(lmax, lmax, randn(ComplexF64, numberOfAlms(lmax))) #[ComplexF64(i) for i in 1:numberOfAlms(lmax)])
else
    test_map = nothing
    test_alm = nothing
end

d_map = DMap{RR}(comm)
MPI.Scatter!(test_map, d_map, comm)
d_alm = DAlm{RR}(comm)
MPI.Scatter!(test_alm, d_alm, comm)

MPI.Barrier(comm)

# TEST ALM2MAP DIRECTION
alm2map!(d_alm, d_map; nthreads = 1)
out_map = deepcopy(test_map)
MPI.Gather!(d_map, out_map)

if crank == root
    Test.@test isapprox(out_map.pixels, alm2map(test_alm, NSIDE).pixels) #test against Healpix transform
end
