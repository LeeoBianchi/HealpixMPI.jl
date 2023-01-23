using Test

@testset "Scatter and Gather map" begin
    cm = @cmd "mpirun -n 3 julia test_MPI_map.jl"
    res = run(cm)
    @test success(res)
end

@testset "Scatter and Gather alm" begin
    cm = @cmd "mpirun -n 3 julia test_MPI_alm.jl"
    res = run(cm)
    @test success(res)
end

@testset "Alm space parallel dot product" begin
    cm = @cmd "mpirun -n 3 julia test_MPI_dot.jl"
    res = run(cm)
    @test success(res)
end

@testset "SHT: alm2map direction" begin
    cm = @cmd "mpirun -n 3 julia test_MPI_a2m.jl"
    res = run(cm)
    @test success(res)
end

@testset "SHT: adjoint direction" begin
    cm = @cmd "mpirun -n 3 julia test_MPI_adj_a2m.jl"
    res = run(cm)
    @test success(res)
end

@testset "Power spectrum functions: alm2cl & synalm!" begin
    cm = @cmd "mpirun -n 3 julia test_MPI_cl.jl"
    res = run(cm)
    @test success(res)
end
