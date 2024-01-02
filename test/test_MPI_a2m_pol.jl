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
    test_alm = [Alm(lmax, lmax, randn(ComplexF64, numberOfAlms(lmax))) for i in 1:3] #[ComplexF64(i) for i in 1:numberOfAlms(lmax)])
    test_out_alm = [Alm(lmax,lmax) for i in 1:3]
else
    out_map_q = nothing
    out_map_u = nothing
    test_map = nothing
    test_alm = nothing
    test_out_alm = nothing
end

#some possible ways to initialize D-objects
d_map_pol = DMap{RR}(comm)
d_alm = DAlm{RR}()
d_alm_pol = DAlm{RR}(comm)

#distribute
MPI.Scatter!(test_map, d_map_pol, comm)
MPI.Scatter!(test_alm, d_alm, d_alm_pol, comm)

MPI.Barrier(comm)
# TEST ALM2MAP DIRECTION
alm2map!(d_alm_pol, d_map_pol; nthreads = 1)

#gather
MPI.Gather!(d_map_pol, out_map_q, 1)
MPI.Gather!(d_map_pol, out_map_u, 2)
MPI.Gather!(d_alm, d_alm_pol, test_out_alm)

if crank == root
    Test.@test isequal(test_out_alm[1].alm, test_alm[1].alm) #test scatter/gather functions
    Test.@test isequal(test_out_alm[2].alm, test_alm[2].alm)
    Test.@test isequal(test_out_alm[3].alm, test_alm[3].alm)
    Test.@test isapprox(out_map_q.pixels, alm2map(test_alm, NSIDE).q.pixels) #test against Healpix transform
    Test.@test isapprox(out_map_u.pixels, alm2map(test_alm, NSIDE).u.pixels) #test against Healpix transform
end
