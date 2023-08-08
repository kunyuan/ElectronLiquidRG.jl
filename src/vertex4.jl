function vertex4(para; neval=1e6, Λgrid=Λgrid(para.kF), n=[-1, 0, 0, -1], l=[0,], filename=nothing)
    ver4, result = Ver4.MC_PH(para; kamp=Λgrid, n=n, l=l, neval=neval, filename=filename, filter=[NoHartree, NoBubble, Proper], channel=[PHr, PHEr, PPr])
    return ver4, result
end

function vertex4_renormalize(para, filename, dz; Fs=fdict[para.rs], Λgrid=Λgrid(para.kF))
    # println("read Fs = $Fs from $filename")
    kF = para.kF
    f = jldopen(filename, "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))

    vuu = Dict()
    vud = Dict()

    for (fi, F) in enumerate(Fs)
        _para = get_para(para, F)
        key = UEG.short(_para)
        kgrid, n, l, ver4  = f[key]
        @assert kgrid ≈ Λgrid

        for p in keys(ver4)
            if haskey(vuu, p) == false
                vuu[p] = MeshArray(Fs, Λgrid; dtype=Complex{Measurement{Float64}})
                vud[p] = MeshArray(Fs, Λgrid; dtype=Complex{Measurement{Float64}})
            end
            vuu[p][fi, :] = ver4[p][1, 1, :]
            vud[p][fi, :] = ver4[p][2, 1, :]
        end
    end

    vuu = CounterTerm.mergeInteraction(vuu)
    vuu = CounterTerm.z_renormalization(para.order, vuu, dz, 1)

    vud = CounterTerm.mergeInteraction(vud)
    vud = CounterTerm.z_renormalization(para.order, vud, dz, 1)

    vuu = [vuu[(o, 0)] for o in 1:para.order]
    vud = [vud[(o, 0)] for o in 1:para.order]
    return vuu, vud
    # for order = 1:order
    # println("up up:", _vuu[2][end, 1])
    # println("up down:", _vud[2][end, 1])
    # end
    # z1[fi, :] = zfactor(sigma[(1, 0, 0)], _para.β)

    # v[1, fi, :] = ve[(1, 0, 0)]
    # return z1
end