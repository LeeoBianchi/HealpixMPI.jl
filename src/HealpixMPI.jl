module HealpixMPI

export make_mvstart_complex, make_mmajor_complex_alm_info, make_mmajor_complex_alm_info!
export DistributedAlm
export DistributedMap
export sharp_alm_info, sharp_geom_info #Libsharp C types
export getlmax, getnm, getmval, getflags, getmvstart, getstride

using Healpix
using MPI
using Libsharp
using Cbinding

#bindings to Libsharp C
let
    incdir = joinpath(libsharp2_jll.artifact_dir, "include")
    libdir = dirname(libsharp2_jll.libsharp2_path)
    c`-std=c99 -march=native -O3 -ffast-math -I$(incdir) -L$(libdir) -lsharp2`
end

const c"ptrdiff_t"  = Cptrdiff_t

c"#include<libsharp2/sharp.h>"J
#####

include("alminfo.jl")
include("alm.jl")
include("map.jl")

end # module
