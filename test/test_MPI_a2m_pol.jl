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
    out_map_q = HealpixMap{Float64, RingOrder}(NSIDE)
    out_map_u = HealpixMap{Float64, RingOrder}(NSIDE)
    test_map = PolarizedHealpixMap{Float64, RingOrder}(NSIDE)
    test_alm3 = [Alm(lmax, lmax, randn(ComplexF64, numberOfAlms(lmax))) for i in 1:3] #[ComplexF64(i) for i in 1:numberOfAlms(lmax)])
    test_alm2 = test_alm3[2:3]
else
    out_map_q = nothing
    out_map_u = nothing
    test_map = nothing
    test_alm2 = nothing
    test_alm3 = nothing
end

d_map = DMap{RR}(comm)
MPI.Scatter!(test_map, d_map, comm)
d_alm = DAlm{RR}(comm)
MPI.Scatter!(test_alm2, d_alm, comm)

MPI.Barrier(comm)

# TEST ALM2MAP DIRECTION
alm2map!(d_alm, d_map; nthreads = 1)

MPI.Gather!(d_map, out_map_q, 1)
MPI.Gather!(d_map, out_map_u, 2)

if crank == root
    #Test.@test isapprox(out_map.i.pixels, alm2map(test_alm, NSIDE).i.pixels)
    Test.@test isapprox(out_map_q.pixels, alm2map(test_alm3, NSIDE).q.pixels) #test against Healpix transform
    Test.@test isapprox(out_map_u.pixels, alm2map(test_alm3, NSIDE).u.pixels) #test against Healpix transform
end
