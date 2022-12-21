module HealpixMPI

export make_mvstart_complex, make_mmajor_complex_alm_info, make_mmajor_complex_alm_info!
export DistributedAlm
export DistributedMap
export sharp_alm_info, sharp_geom_info #Libsharp C types
export getlmax, getnm, getmval, getflags, getmvstart, getstride

using Healpix
using MPI
using Libsharp

#bindings to Libsharp C
using CBinding
c`-std=c99 -L$/usr/local/lib/libsharp2.so -lsharp2`;
const c"ptrdiff_t"  = Cptrdiff_t
c"#include<libsharp2/sharp.h>"J
#####

include("alminfo.jl")
include("alm.jl")
include("map.jl")

end # module
