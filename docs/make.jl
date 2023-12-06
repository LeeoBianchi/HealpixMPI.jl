push!(LOAD_PATH,"../src/")

using HealpixMPI
using Documenter
using MPI
using LinearAlgebra
using Healpix

makedocs(
         sitename = "HealpixMPI.jl",
         modules  = [HealpixMPI],
         pages=[
                "Introduction" => "index.md"
                "Distributed Classes" => "distribute.md"
                "Spherical Harmonics" => "sht.md"
                #"Miscellanea" => "misc.md"
               ])

deploydocs(
    repo="github.com/LeeoBianchi/HealpixMPI.jl.git",
)
