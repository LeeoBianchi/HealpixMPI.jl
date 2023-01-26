push!(LOAD_PATH,"../src/")

using HealpixMPI
using Documenter

makedocs(
         sitename = "HealpixMPI.jl",
         modules  = [HealpixMPI],
         pages=[
                "Home" => "index.md"
               ])

deploydocs(;
    repo="github.com/LeeoBianchi/HealpixMPI.jl",
)