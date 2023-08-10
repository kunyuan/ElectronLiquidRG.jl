using ElectronLiquid
import ElectronLiquidRG as RG
using Measurements
using GreenFunc
using JLD2
using Printf

const rs = 1.0
const beta = 25.0
const mass2 = 0.01
const order = 1

const para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order+1)

const Fs = RG.fdict[para.rs]
const Λgrid= RG.Λgrid(para.kF)

function get_z()
    dzi, dmu, dz = RG.zCT(para, "data/sigma.jld2"; Fs=Fs, Λgrid=Λgrid)

    z1 = dz[1]

    println(z1[end, :])
    println(z1.mesh[1][end])
    println(z1.mesh[2][1])

    return dz
end

function get_ver3()
    f = jldopen("data/ver3.jld2", "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))
    ver3 = MeshArray(Fs, Λgrid; dtype=Complex{Measurement{Float64}})

    for (fi, F) in enumerate(Fs)
        _para = RG.get_para(para, F)
        key = UEG.short(_para)
        kgrid, _ver3 = f[key]
        ver3[fi, :] = _ver3
        @assert kgrid ≈ Λgrid
    end
    return ver3
end

function get_ver4(dz)
    vuu, vud = RG.vertex4_renormalize(para, "data/ver4.jld2", dz; Fs=Fs, Λgrid=Λgrid)

    return vuu, vud
end

function print_ver3(ver3, fi = size(ver3)[1]; nsample = 5)
    kF = para.kF
    Fs = ver3.mesh[1]
    kgrid = ver3.mesh[2]
    step = Int(ceil(length(kgrid)/nsample))
    kidx = [i for i in 1:step:length(kgrid)]
    # fi = length(Fs)
    println("Fs = $(Fs[fi]) at index $fi")
    printstyled(@sprintf("%12s    %24s\n",
            "k/kF", "ver3"), color=:yellow)
    for ki in kidx
        @printf("%12.6f    %24s\n", kgrid[ki] / kF, "$(ver3[fi, ki])")
    end
end


function print_ver4(vuu, vud, fi = size(vuu[1])[1]; nsample = 5)
    kF = para.kF
    # Fs = RG.fdict[para.rs]
    # kgrid= RG.Λgrid(para.kF)
    sample = vuu[1]
    Fs = sample.mesh[1]
    kgrid = sample.mesh[2]
    step = Int(ceil(length(kgrid)/nsample))
    kidx = [i for i in 1:step:length(kgrid)]
    println(kidx)
    # fi = length(Fs)
    println("Fs = $(Fs[fi]) at index $fi")
    for p in eachindex(vuu)
    # for p in keys(vuu)
        # head = ["k/kF", "uu", "ud", "symmetric", "asymmetric"]

        printstyled("Order = $p\n", color=:red)
        printstyled(@sprintf("%12s    %24s    %24s    %24s    %24s     %24s\n",
                "k/kF", "uu", "ud", "symmetric", "asymmetric", "p: $p"), color=:yellow)
        for ki in kidx
            factor = 1.0
            d1, d2 = real(vuu[p][fi, ki]) , real(vud[p][fi, ki])
            # s, a = (d1 + d2) / 2.0, (d1 - d2) / 2.0
            s, a = Ver4.ud2sa(d1, d2)
            @printf("%12.6f    %24s    %24s    %24s    %24s\n", kgrid[ki] / kF, "$d1", "$d2", "$s", "$a")
        end
    end

    printstyled("summed\n", color=:red)
    printstyled(@sprintf("%12s    %24s    %24s    %24s    %24s\n",
            "k/kF", "uu", "ud", "symmetric", "asymmetric"), color=:yellow)
    for ki in kidx
        _vuu = sum(vuu)
        _vud = sum(vud)
        d1, d2 = real(_vuu[fi, ki]) , real(_vud[fi, ki]) 
        # s, a = (d1 + d2) / 2.0, (d1 - d2) / 2.0
        s, a = Ver4.ud2sa(d1, d2)
        @printf("%12.6f    %24s    %24s    %24s    %24s\n", kgrid[ki] / kF, "$d1", "$d2", "$s", "$a")
    end
end

a(Fs, k, vs) = RG.linear_interp(Fs, vs, k)

b(Fs, k, ver3) = RG.linear_interp(Fs, vs, ver3)/para.NF 

const c = para.me/8/π/para.NF # ~0.2 for rs=1

function solve_RG(vuu, vud, ver3)
    a = (vuu+vuv)/2.0
    b = ver3

end

dz = get_z()
ver3 = get_ver3()
print_ver3(ver3; nsample =10)
vuu, vud = get_ver4(dz)
print_ver4(vuu, vuv, 1; nsample=10)
print_ver4(vuu, vuv; nsample = 10)

vs = (vuu + vud) / 2.0
ver3 = 