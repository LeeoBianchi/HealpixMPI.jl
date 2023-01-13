include("sht.jl")

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

d_map = DistributedMap()
MPI.Scatter!(test_map, d_map, comm)
d_alm = DistributedAlm()
MPI.Scatter!(test_alm, d_alm, comm)

alm2map!(d_alm, d_map, nthreads = 1)
#FIXME: check rstart array
alm2map(test_alm, 2)

get_nm_RR(5, 0, 1)

Array{Float64, 3}()
