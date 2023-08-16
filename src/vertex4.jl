function vertex4(para; neval=1e6, Λgrid=Λgrid(para.kF), n=[-1, 0, 0, -1], l=[0,], filename=nothing)
    kamp2 = [para.kF for i in Λgrid]
    ver4, result = Ver4.MC_PH(para; kamp=Λgrid, kamp2=kamp2, n=n, l=l, neval=neval, filename=filename, filter=[NoHartree, NoBubble, Proper], channel=[PHr, PHEr, PPr])
    # ver4, result = Ver4.MC_PH(para; kamp=Λgrid, n=n, l=l, neval=neval, filename=filename, filter=[NoHartree, NoBubble, Proper], channel=[PHr, PHEr, PPr])
    return ver4, result
end

function vertex4_renormalize(para, filename, dz; Fs=fdict[para.rs], Λgrid=SparseΛgrid(para.kF))
    # println("read Fs = $Fs from $filename")
    kF = para.kF
    f = jldopen(filename, "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))

    vuu = Dict()
    vud = Dict()

    for (fi, F) in enumerate(Fs)
        _para = get_para(para, F)
        key = UEG.short(_para)
        kgrid, n, l, ver4 = f[key]
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
    for k in keys(vuu)
        println(vuu[k][end, 1])
        println(vud[k][end, 1])
    end

    vuu = CounterTerm.mergeInteraction(vuu)
    vuu = CounterTerm.z_renormalization(para.order, vuu, dz, 2)

    vud = CounterTerm.mergeInteraction(vud)
    vud = CounterTerm.z_renormalization(para.order, vud, dz, 2)

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

"""
u*G0*G0*u ladder diagram. The (up, up, up, up) is zero and (up, up, down, down) is non-zero 
"""
function c_coeff_pp(para, kamp=para.kF, kamp2=para.kF)
    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 0.001, 32)
    qs = [sqrt(kamp^2 + kamp2^2 - 2 * cos(θ) * kamp * kamp2) for θ in θgrid.grid]

    Wp = zeros(ComplexF64, length(qs))
    for (qi, q) in enumerate(qs)
        Wp[qi] = ElectronGas.Polarization.Ladder0_FiniteTemp(q, 0, para)
    end

    vud = Interp.integrate1D(Wp .* sin.(θgrid.grid), θgrid) / 2 / para.NF
    return 0, -real(vud)
end

"""
u*Pi_0*u particle-hole exchange diagram. The (up, up, up, up) and (up, up, down, down) have the same weight
"""
function c_coeff_phe(para, kamp=para.kF, kamp2=para.kF)
    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 0.001, 32)
    qs = [sqrt(kamp^2 + kamp2^2 - 2 * cos(θ) * kamp * kamp2) for θ in θgrid.grid]

    Wp = zeros(ComplexF64, length(qs))
    for (qi, q) in enumerate(qs)
        Wp[qi] = ElectronGas.Polarization.Polarization0_FiniteTemp(q, 0, para)
    end

    vud = Interp.integrate1D(Wp .* sin.(θgrid.grid), θgrid) / 2 / para.NF
    vuu = vud
    return real(vuu), real(vud)
end



