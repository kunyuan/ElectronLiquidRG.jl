using ElectronLiquid
import ElectronLiquidRG as RG
using JLD2
using Printf

const rs = 5.0
const beta = 25.0
const mass2 = 0.001
const order = 1

function get_z()
    para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order)
    Fs = RG.fdict[para.rs]
    Λgrid= RG.Λgrid(para.kF)
    println("Fs = $Fs")
    println("Λgrid = $Λgrid")

    dzi, dmu, dz = RG.zCT(para, "data/sigma.jld2"; Fs=Fs, Λgrid=Λgrid)

    z1 = dz[1]

    println(z1[end, :])
    println(z1.mesh[1][end])
    println(z1.mesh[2][1])

    return dz
end

function get_ver4(dz)
    para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order+1)
    Fs = RG.fdict[para.rs]
    Λgrid= RG.Λgrid(para.kF)
    println("Fs = $Fs")
    println("Λgrid = $Λgrid")
    vuu, vud = RG.vertex4_renormalize(para, "data/ver4.jld2", dz; Fs=Fs, Λgrid=Λgrid)
    kgrid = Λgrid[1:10:end]
    kF = para.kF

    fi = length(Fs)
    println("Fs = $(Fs[fi])")
    for p in eachindex(vuu)
    # for p in keys(vuu)
        # head = ["k/kF", "uu", "ud", "symmetric", "asymmetric"]

        printstyled("Order = $p\n", color=:red)
        printstyled(@sprintf("%12s    %24s    %24s    %24s    %24s     %24s\n",
                "k/kF", "uu", "ud", "symmetric", "asymmetric", "p: $p"), color=:yellow)
        for (ki, k) in enumerate(kgrid)
            factor = 1.0
            d1, d2 = real(vuu[p][fi, ki]) , real(vud[p][fi, ki])
            # s, a = (d1 + d2) / 2.0, (d1 - d2) / 2.0
            s, a = Ver4.ud2sa(d1, d2)
            @printf("%12.6f    %24s    %24s    %24s    %24s\n", k / kF, "$d1", "$d2", "$s", "$a")
        end
    end

    printstyled("summed\n", color=:red)
    printstyled(@sprintf("%12s    %24s    %24s    %24s    %24s\n",
            "k/kF", "uu", "ud", "symmetric", "asymmetric"), color=:yellow)
    for (ki, k) in enumerate(kgrid)
        _vuu = vuu[1] + vuu[2]
        _vud = vud[1] + vud[2]
        d1, d2 = real(_vuu[fi, ki]) , real(_vud[fi, ki]) 
        # s, a = (d1 + d2) / 2.0, (d1 - d2) / 2.0
        s, a = Ver4.ud2sa(d1, d2)
        @printf("%12.6f    %24s    %24s    %24s    %24s\n", k / kF, "$d1", "$d2", "$s", "$a")
    end
end

dz = get_z()
get_ver4(dz)