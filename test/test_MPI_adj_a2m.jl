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
    test_map = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(NSIDE)])
    test_alm = Alm(lmax, lmax, [ComplexF64(i) for i in 1:numberOfAlms(lmax)])
else
    test_map = nothing
    test_alm = nothing
end

d_map = DMap{RR}(comm)
MPI.Scatter!(test_map, d_map, comm)
d_alm = DAlm{RR}(comm)
MPI.Scatter!(test_alm, d_alm, comm)

MPI.Barrier(comm)

# TEST ADJOINT DIRECTION:
adjoint_alm2map!(d_map, d_alm; nthreads = 1)
out_alm = deepcopy(test_alm)
out_alm2 = deepcopy(test_alm)
MPI.Gather!(d_alm, out_alm)


if crank == root
    adjoint_alm2map!(test_map, out_alm2)
    Test.@test isapprox(out_alm.alm, out_alm2.alm)
end
