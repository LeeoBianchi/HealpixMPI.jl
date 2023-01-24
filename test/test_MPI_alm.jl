using Healpix
using MPI
using Test

include("../src/alm.jl")

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

test_alm_all = Alm(5, 5, [ComplexF64(i) for i in 1:numberOfAlms(5)])
res_alm_all = Alm(5,5)

d_alm = DistributedAlm()

MPI.Scatter!(test_alm, d_alm, comm)

MPI.Gather!(d_alm, res_alm)

MPI.Allgather!(d_alm, res_alm_all, clear=true)

MPI.Barrier(comm)

if crank == root
    Test.@test test_alm.alm == res_alm.alm
else
    Test.@test res_alm == nothing
end
res_alm_all.alm
Test.@test res_alm_all.alm == test_alm_all.alm #FIXME: check why allgather doesn't work on 1 task
