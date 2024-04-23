using MPI
using Healpix
using HealpixMPI
using BenchmarkTools
using DelimitedFiles
MPI.Init()

comm = MPI.COMM_WORLD
global crank = MPI.Comm_rank(comm)
global csize = MPI.Comm_size(comm)
root = 0

NSIDE = 4096
lmax = 3*NSIDE - 1

#initialize some Healpix data
if crank == root
    test_map = HealpixMap{Float64, RingOrder}(NSIDE)
    test_alm = Alm(lmax, lmax, [ComplexF64(i) for i in 1:numberOfAlms(lmax)])
else
    test_map = nothing
    test_alm = nothing
end

#distribute it into HealpixMPI types
d_map = DMap{RR}(comm)
MPI.Scatter!(test_map, d_map, comm)
d_alm = DAlm{RR}(comm)
MPI.Scatter!(test_alm, d_alm, comm)

#auxiliary optional legeandre coeffs arrays for more efficient SHTs
aux_alm_leg = Array{ComplexF64,3}(undef, (length(d_alm.info.mval), numOfRings(d_map.info.nside), 1)) # loc_nm * tot_nr
aux_map_leg = Array{ComplexF64,3}(undef, d_alm.info.mmax+1, length(d_map.info.rings), 1)

NTs = [2, 4, 8 , 16, 32] #different number of threads to run the benchmarks with.
ba2m = Vector{Float64}(undef, length(NTs)) #result arrays
bm2a = Vector{Float64}(undef, length(NTs)) #result arrays
#run benchmarks
for i in 1:length(NTs)
	println("alm2map benchmark for $csize MPI tasks and $(NTs[i]) C++ threads ...")
	global NT = NTs[i]
    b = @benchmark alm2map!(d_alm, d_map, aux_alm_leg, aux_map_leg; nthreads = NT) samples = 20 seconds = 600 evals = 1
    ba2m[i] = minimum(b.times) * NT * csize / 1e9 #in seconds
	println("adjoint_alm2map benchmark for $csize MPI tasks and $(NTs[i]) C++ threads ...")
    b = @benchmark adjoint_alm2map!(d_map, d_alm, aux_map_leg, aux_alm_leg; nthreads = NT) samples = 20 seconds = 600 evals = 1
    bm2a[i] = minimum(b.times) * NT * csize / 1e9 #in seconds
end
#print results
if crank == root
    outfile = "HpixMPI_a2m_full_Nside$(NSIDE)_$(csize)MPI_$(NT)threads.txt"
	writedlm(outfile, ba2m)
	outfile = "HpixMPI_adj_full_Nside$(NSIDE)_$(csize)MPI_$(NT)threads.txt"
	writedlm(outfile, bm2a)
end