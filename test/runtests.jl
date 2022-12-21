using Main.HealpixMPI
using Test

###Then remove:
using Healpix
using Libsharp
include("/home/leoab/OneDrive/UNI/Tesi_Oslo/newfunc.jl")
#include("/home/leoab/OneDrive/UNI/Tesi_Oslo/newfunc_MPI_2.jl")
include("/home/leoab/OneDrive/UNI/HealpixMPI/src/alm.jl")
include("/home/leoab/OneDrive/UNI/HealpixMPI/src/map.jl")
###

function make_subset_healpix_geom_info(
    nside::Integer,
    stride::Integer,
    rings::AbstractArray{T} #FIXME: add overload for weights in input if needed!
    ) where T <: Integer

    geom_info_ptr = Ref{Ptr{Cvoid}}()
    nrings = length(rings)

    rings_cint = [Cint(x) for x in rings]
    ccall(
        (:sharp_make_subset_healpix_geom_info, libsharp2),
        Cvoid,
        (Cint, Cint, Cint, Ref{Cint}, Ref{Cdouble}, Ref{Ptr{Cvoid}}),
        nside, stride, nrings, rings_cint, Ptr{Cdouble}(C_NULL), geom_info_ptr,
    )
    GeomInfo(geom_info_ptr[])
end

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    test_map = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(2)])
    test_alm = Alm(5, 5, [ComplexF64(i) for i in 1:numberOfAlms(5)])
    res_map = HealpixMap{Float64, RingOrder}(zeros(nside2npix(2)))
    res_alm = Alm(5,5)
else
    test_map = nothing
    test_alm = nothing
    res_alm = nothing
    res_map = nothing
end

d_map = DistributedMap()
MPI.Scatter!(test_map, d_map, :RR, comm)

d_alm = DistributedAlm()
MPI.Scatter!(test_alm, d_alm, :RR, comm)

MPI.Gather!(d_map, res_map, :RR, comm, clear=true)

MPI.Gather!(d_alm, res_alm, :RR, comm, clear=false)

MPI.Barrier(comm)

if crank == root
    @test test_alm.alm == res_alm.alm
    @test test_map == res_map
end

##test distributed alm space dot
standardot = test_alm⋅test_alm
distridot = d_alm⋅d_alm
@test standardot == distridot

##
