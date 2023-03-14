push!(LOAD_PATH,"../src/")

using HealpixMPI
using Documenter

makedocs(
         sitename = "HealpixMPI.jl",
         modules  = [HealpixMPI],
         pages=[
                "Introduction" => "index.md"
                "Distributed Classes" => "distribute.md"
                "Spherical Harmonics" => "sht.md"
               ])

deploydocs(;
    repo="github.com/LeeoBianchi/HealpixMPI.jl",
)
