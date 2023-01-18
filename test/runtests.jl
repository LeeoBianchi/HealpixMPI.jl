using Test

@testset "Scatter and Gather map" begin
    cm = @cmd "mpirun -n 4 julia test_MPI_map.jl"
    res = run(cm)
    @test success(res)
end

@testset "Scatter and Gather alm" begin
    cm = @cmd "mpirun -n 4 julia test_MPI_alm.jl"
    res = run(cm)
    @test success(res)
end

@testset "SHT: alm2map & adjoint" begin
    cm = @cmd "mpirun -n 2 julia test_MPI_alm.jl"
    res = run(cm)
    @test success(res)
end
