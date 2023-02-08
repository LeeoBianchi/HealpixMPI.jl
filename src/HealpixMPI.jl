module HealpixMPI

export Strategy, RR
export AlmInfoMPI, DistributedAlm
export GeomInfoMPI, DistributedMap
export make_mstart_complex, get_nm_RR, get_mval_RR, get_m_tasks_RR
export ring2theta, get_equator_idx, get_ring_pixels, get_nrings_RR, get_rindexes_RR
export communicate_alm2map!, communicate_map2alm!

import Healpix
import MPI

include("strategy.jl")
include("alm.jl")
include("map.jl")
include("sht.jl")
include("cl.jl")
include("tools.jl")

end # module
