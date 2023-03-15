using Test
using MPI

testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "test_")
testfiles = sort(filter(istest, readdir(testdir)))
nprocs = clamp(Sys.CPU_THREADS, 2, 4)

@testset "$f" for f in testfiles
    mpiexec() do mpirun
        cmd(n) = `$mpirun -n $n $(Base.julia_cmd()) --startup-file=no $(joinpath(testdir, f))`
        res = run(cmd(nprocs))
        @test success(res)
    end
end
