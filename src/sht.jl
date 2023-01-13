include("/home/leoab/OneDrive/UNI/ducc/julia/ducc0.jl") #FIXME: replace with proper binding to Ducc0

include("alm.jl")
include("map.jl")

import Healpix: alm2map! #adjoint_alm2map!, when it will be added in Healpix.jl

function communicate_alm2map(args)
    body
end

function alm2map!()
    body
end

function communicate_map2alm(args)
    body
end

function adjoint_alm2map!(args)
    body
end
