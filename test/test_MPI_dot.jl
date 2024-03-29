using Test
using MPI
using Healpix
include("../src/HealpixMPI.jl")
using Main.HealpixMPI
using LinearAlgebra
#################################################################

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    test_alm = Alm(5, 5, [ComplexF64(i) for i in 1:numberOfAlms(5)])
    res_alm = Alm(5,5)
else
    test_alm = nothing
    res_alm = nothing
end

d_alm = DAlm{RR}()

MPI.Scatter!(test_alm, d_alm, comm)

MPI.Barrier(comm)

Test.@test d_alm ⋅ d_alm == 6531.0
