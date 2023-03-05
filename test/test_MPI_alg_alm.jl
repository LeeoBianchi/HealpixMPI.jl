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

if crank == root
    test_alm = Alm(10, 10, randn(ComplexF64, numberOfAlms(10)))
else
    test_alm = nothing
end

d_alm = DistributedAlm{RR}()

MPI.Scatter!(test_alm, d_alm, comm)

#TEST ALGEBRA
d_alm2 = 2*d_alm

Test.@test (d_alm + d_alm).alm == d_alm2.alm
Test.@test (d_alm2/2).alm == (d_alm2 - d_alm).alm
Test.@test isapprox((d_alm * d_alm / d_alm).alm, d_alm.alm)

fl = Vector{Float64}(1:11)
Test.@test isapprox(((fl*d_alm)/fl).alm, d_alm.alm)

d_alm2 = d_alm*fl
ref_alm = deepcopy(test_alm)
MPI.Gather!(d_alm2, ref_alm)
if crank == root
    Test.@test isapprox(ref_alm.alm, (test_alm*fl).alm)
end
