using Healpix
using MPI
using Test
include("../src/sht.jl")

MPI.Init()

comm = MPI.COMM_WORLD
crank = MPI.Comm_rank(comm)
csize = MPI.Comm_size(comm)
root = 0

if crank == root
    test_map = HealpixMap{Float64, RingOrder}([Float64(i) for i in 1:nside2npix(2)])
    test_alm = Alm(5, 5, [ComplexF64(i) for i in 1:numberOfAlms(5)])
else
    test_map = nothing
    test_alm = nothing
end

d_map = DistributedMap(comm)
MPI.Scatter!(test_map, d_map, comm)
d_alm = DistributedAlm(comm)
MPI.Scatter!(test_alm, d_alm, comm)

MPI.Barrier(comm)

# TEST ALM2MAP DIRECTION

alm2map!(d_alm, d_map; nthreads = 1)
out_map = deepcopy(test_map)
MPI.Gather!(d_map, out_map)

if crank == root
    Test.@test isapprox(out_map.pixels, alm2map(test_alm, 2).pixels) #replace @show with test
end

# ADJOINT DIRECTION:
adjoint_alm2map!(d_map, d_alm; nthreads = 1)
out_alm = deepcopy(test_alm)
out_alm2 = deepcopy(test_alm)
MPI.Gather!(d_alm, out_alm)

#FIXME: remove when it will be available in Healpix.jl
function adjoint_alm2map!(
    map::HealpixMap{Float64,RingOrder,Array{Float64,1}},
    alm::Alm{ComplexF64,Array{ComplexF64,1}}
)
    geom_info = Libsharp.make_healpix_geom_info(map.resolution.nside, 1)
    alm_info = Libsharp.make_triangular_alm_info(alm.lmax, alm.mmax, 1)
    Libsharp.sharp_execute!(
        Libsharp.SHARP_Yt,
        0,
        [alm.alm],
        [map.pixels],
        geom_info,
        alm_info,
        Libsharp.SHARP_DP,
    )
end

if crank == root
    adjoint_alm2map!(test_map, out_alm2)
    Test.@test isapprox(out_alm.alm, out_alm2.alm)
end
