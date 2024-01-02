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

test_alm = Alm(5, 5, [ComplexF64(i) for i in 1:numberOfAlms(5)])

d_alm = DAlm{RR}()

MPI.Scatter!(test_alm, d_alm, comm)

MPI.Barrier(comm)

@test isapprox(alm2cl(d_alm), Healpix.alm2cl(test_alm))

cl = Vector{Float64}(0:6)
synalm!(cl, d_alm, comp = 1)

if crank == root
    #task 0 has m = 0
    @test d_alm.alm[1] == 0.0 + 0.0im #\ell=0 should be 0 because of our cl's
    @test imag.(d_alm.alm[1:5]) == zeros(Float64, 5) #field should be always real
    @test count(i->(i==0.00000000000e+00), real.(d_alm.alm[2:end])) == 0 #none of the other real part of alms should be exactly 0.
end
@test count(i->(i==0.00000000000e+00), imag.(d_alm.alm[7:end])) == 0 #none of the other imag part of alms from lmax on should be exactly 0.
