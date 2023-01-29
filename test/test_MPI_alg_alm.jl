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
