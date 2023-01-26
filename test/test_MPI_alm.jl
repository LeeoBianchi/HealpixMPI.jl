using Healpix
import Test

include("../src/alm.jl")

#################################################################

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    test_alm = Alm(10, 10, [ComplexF64(i) for i in 1:numberOfAlms(10)])
    res_alm = Alm(10,10)
else
    test_alm = nothing
    res_alm = nothing
end

test_alm_all = Alm(10, 10, [ComplexF64(i) for i in 1:numberOfAlms(10)])
res_alm_all = Alm(10,10)

d_alm = DistributedAlm()

MPI.Scatter!(test_alm, d_alm, comm)

MPI.Gather!(d_alm, res_alm)

MPI.Allgather!(d_alm, res_alm_all, clear=false)

MPI.Barrier(comm)

#TEST SCATTER/GATHER
if crank == root
    Test.@test test_alm.alm == res_alm.alm
else
    Test.@test res_alm === nothing
end
Test.@test res_alm_all.alm == test_alm_all.alm #FIXME: check why allgather doesn't work on 1 task