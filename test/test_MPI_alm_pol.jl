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
lmax = 10

if crank == root
    test_alm = [Alm(lmax, lmax, [ComplexF64(i) for i in 1:numberOfAlms(lmax)]) for i in 1:3]
    res_alm = [Alm(lmax,lmax) for i in 1:3]
else
    test_alm = nothing
    res_alm = nothing
end

test_alm_all = [Alm(lmax, lmax, [ComplexF64(i) for i in 1:numberOfAlms(lmax)]) for i in 1:3]
res_alm_all = [Alm(lmax,lmax) for i in 1:3]

d_alm = DAlm{RR}(comm)
d_alm_pol = DAlm{RR}(comm)

MPI.Scatter!(test_alm, d_alm, d_alm_pol, comm)

MPI.Gather!(d_alm, d_alm_pol, res_alm)
MPI.Allgather!(d_alm, d_alm_pol, res_alm_all, clear = true)

MPI.Barrier(comm)

#TEST SCATTER/GATHER
for i in 1:3
    if crank == root
        Test.@test test_alm[i].alm == res_alm[i].alm
    else
        Test.@test res_alm === nothing
    end
    Test.@test res_alm_all[i].alm == test_alm_all[i].alm
end
