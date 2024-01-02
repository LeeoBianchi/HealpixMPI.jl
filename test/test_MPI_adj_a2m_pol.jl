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
    test_map = PolarizedHealpixMap{Float64, RingOrder}(
        randn(Float64, nside2npix(NSIDE)),
        randn(Float64, nside2npix(NSIDE)),
        randn(Float64, nside2npix(NSIDE)))
    test_out_map = PolarizedHealpixMap{Float64, RingOrder}(NSIDE)
    test_alm3 = [Alm(lmax, lmax) for i in 1:3] #[ComplexF64(i) for i in 1:numberOfAlms(lmax)])
    test_alm2 = test_alm3[2:3]
else
    out_map_q = nothing
    out_map_u = nothing
    test_map = nothing
    test_out_map = nothing
    test_alm2 = nothing
    test_alm3 = nothing
end

#some possible ways to initialize the D-objects
d_map = DMap{RR}()
d_map_pol = DMap{RR}(comm)
d_alm_pol = DAlm{RR}(comm)

#distribute
MPI.Scatter!(test_map, d_map, d_map_pol, comm)
MPI.Scatter!(test_alm2, d_alm_pol, comm)

MPI.Barrier(comm)

# TEST ALM2MAP DIRECTION
adjoint_alm2map!(d_map_pol, d_alm_pol; nthreads = 1)

MPI.Gather!(d_alm_pol, test_alm2, clear = true)
MPI.Gather!(d_map, d_map_pol, test_out_map, clear = true)

if crank == root
    adjoint_alm2map!(test_map, test_alm3)
    Test.@test isapprox(test_alm2[1].alm, test_alm3[2].alm) #test against Healpix transform
    Test.@test isapprox(test_alm2[2].alm, test_alm3[3].alm) #shifted to ignore T component.
    Test.@test isequal(test_map.i.pixels, test_out_map.i.pixels) #test against Healpix transform
    Test.@test isequal(test_map.q.pixels, test_out_map.q.pixels) #test against Healpix transform
    Test.@test isequal(test_map.u.pixels, test_out_map.u.pixels) #test against Healpix transform
end
