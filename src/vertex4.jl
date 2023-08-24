function vertex4(para; neval=1e6, Λgrid=Λgrid(para.kF), n=[-1, 0, 0, -1], l=[0,], filename=nothing)
    kamp2 = [para.kF for i in Λgrid]
    ver4, result = Ver4.MC_PH(para; kamp=Λgrid, kamp2=kamp2, n=n, l=l, neval=neval, filename=filename, filter=[NoHartree, NoBubble, Proper], channel=[PHr, PHEr, PPr])
    # ver4, result = Ver4.MC_PH(para; kamp=Λgrid, n=n, l=l, neval=neval, filename=filename, filter=[NoHartree, NoBubble, Proper], channel=[PHr, PHEr, PPr])
    return ver4, result
end

function vertex4_renormalize(para, filename, dz, dz2=dz; Fs=fdict[para.rs], Λgrid=SparseΛgrid(para.kF))
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
    # for k in keys(vuu)
    #     println(vuu[k][end, 1])
    #     println(vud[k][end, 1])
    # end

    vuu = CounterTerm.mergeInteraction(vuu)
    vuu = CounterTerm.z_renormalization(para.order, vuu, dz, 2) #left leg renormalization
    # vuu = CounterTerm.z_renormalization(para.order, vuu, dz2, 1) #right leg renormalization

    vud = CounterTerm.mergeInteraction(vud)
    vud = CounterTerm.z_renormalization(para.order, vud, dz, 2) #left leg renormalization
    # vud = CounterTerm.z_renormalization(para.order, vud, dz2, 1) #right leg renormalization

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
-u*G0*G0*u + #Λ∞ ladder diagram subtracting the constant term. The (up, up, up, up) is zero and (up, up, down, down) is non-zero 
"""
function c_coeff_pp(para, kamp=para.kF, kamp2=para.kF)
    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 0.001, 32)
    qs = [sqrt(kamp^2 + kamp2^2 - 2 * cos(θ) * kamp * kamp2) for θ in θgrid.grid]

    Wp = zeros(ComplexF64, length(qs))
    for (qi, q) in enumerate(qs)
        Wp[qi] = ElectronGas.Polarization.Ladder0_FiniteTemp(q, 0, para)
    end

    vud = Interp.integrate1D(Wp .* sin.(θgrid.grid), θgrid) / 2
    return 0, -real(vud)
end

"""
u*Pi_0*u particle-hole exchange diagram. 
1. (up, up, up, up): the internal bubble is spin down, and there is a Fermi loop contributing a minus sign
2. (up, up, down, down): the internal bubble is up and down, and there is no closed Fermi loop, so there is no additional sign.

For a reference of the diagram sign, according to ElectronLiquid.Ver4.MC_PH subroutine, for rs=1, mass=10.0, Yukawa interaction
 # order                 upup          updown
    1 (exchange)        0.152(1)        0.0 
    2 (PHEr)            0.0088(1)    -0.0119(2)
    2 (PPr)             0.002083     0.02198(42) 

WARNING 1: Following the convention of the RG application, we need to add an additional minus to the above results.
WARNING 2: The main contributions in the PPr diagrams are from the UV, so it should be compared with Λ∞ - uG0G0u where Λ∞ needs to be fitted.
"""
function c_coeff_phe(para, kamp=para.kF, kamp2=para.kF)
    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 0.001, 32)
    qs = [sqrt(kamp^2 + kamp2^2 - 2 * cos(θ) * kamp * kamp2) for θ in θgrid.grid]

    Wp = zeros(ComplexF64, length(qs))
    for (qi, q) in enumerate(qs)
        Wp[qi] = ElectronGas.Polarization.Polarization0_FiniteTemp(q, 0, para)
    end

    vud = Interp.integrate1D(Wp .* sin.(θgrid.grid), θgrid) / 2
    vuu = vud
    return real(vuu), -real(vud)
end


#Particle-particle diagram, order = 2
# sign = (-1)^order = 1
#
# kamp,up   kamp2,down
#t2 |-- KO ---| t2/t3
#   |         |
# up^         ^down
#   |         |
# 0 |-- KO ---| 0/t1
#kamp,up  kamp2,down
# with Yukawa interaction mass2 = 0.1, we use Ver4.MC_PH to calculate the PPr diagram, where the leading order diagram should be the same as the particle-particle diagram, we obtain 
# order                upup            updown
#   1                 0.328(4)          0.0
#   2                 0.465(16)         0.661(22)
# therefore, we expect PP to be positively defined
function _ver4PP(vars, config)
    varK, varT, varX, varN = vars
    R, Theta, Phi = varK
    para, kamp, kamp2 = config.userdata

    t1, t2, t3 = varT[2], varT[3], varT[4]
    r = R[1] / (1 - R[1])
    θ, ϕ = Theta[1], Phi[1]
    x = varX[1]
    ki = varN[1]
    kl, kr = kamp[ki], kamp2[ki]


    q = [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

    k1 = [kl + q[1], q[2], q[3]]
    k2 = [kr * x - q[1], kr * sqrt(1 - x^2) - q[2], -q[3]]

    ek1 = (dot(k1, k1) - para.kF^2) / 2 / para.me
    ek2 = (dot(k2, k2) - para.kF^2) / 2 / para.me

    g1 = Spectral.kernelFermiT(t2, ek1, para.β)
    g2 = Spectral.kernelFermiT(t2, ek2, para.β)
    g3 = Spectral.kernelFermiT(t2 - t1, ek2, para.β)
    g4 = Spectral.kernelFermiT(t3, ek2, para.β)
    g5 = Spectral.kernelFermiT(t3 - t1, ek2, para.β)

    qd = sqrt(dot(q, q))
    vq1 = UEG.interactionStatic(para, qd, 0.0, t1)
    wq1 = UEG.interactionDynamic(para, qd, 0.0, t1)

    vq2 = UEG.interactionStatic(para, qd, t2, t3)
    wq2 = UEG.interactionDynamic(para, qd, t2, t3)

    qe = [-kl + kr * x - q[1], kr * sqrt(1 - x^2) - q[2], -q[3]]
    qe = sqrt(dot(qe, qe))
    vq2e = UEG.interactionStatic(para, qe, t2, t3)
    wq2e = UEG.interactionDynamic(para, qe, t2, t3)

    # vq0 = -4π *para.e0^2 / (qd^2+para.mass2) /para.β 
    # @assert vq ≈ vq0 "vq=$vq, vq0=$vq0, qd=$qd, t1=$t1"
    # wq = 0.0
    # println(t1)

    phase1 = phase((0.0, t2, 0.0, t2), -1, 0, 0, para.β)
    phase2 = phase((0.0, t2, t1, t2), -1, 0, 0, para.β)

    phase3 = phase((0.0, t2, 0.0, t3), -1, 0, 0, para.β)
    phase4 = phase((0.0, t2, t1, t3), -1, 0, 0, para.β)

    phase3e = phase((0.0, t3, 0.0, t2), -1, 0, 0, para.β)
    phase4e = phase((0.0, t3, t1, t2), -1, 0, 0, para.β)

    factor = r^2 * sin(θ) / (1 - R[1])^2 / (2π)^3 * para.NF
    factor /= 2 # angle average with l=0

    wd = g1 * g2 * vq1 * (vq2) * phase1
    wd += g1 * g3 * wq1 * (vq2) * phase2
    wd += g1 * g4 * vq1 * (wq2) * phase3
    wd += g1 * g5 * wq1 * (wq2) * phase4
    wd *= factor

    we = g1 * g2 * vq1 * (-vq2e) * phase1
    we += g1 * g3 * wq1 * (-vq2e) * phase2
    we += g1 * g4 * vq1 * (-wq2e) * phase3e
    we += g1 * g5 * wq1 * (-wq2e) * phase4e
    we *= factor

    # return wd + we / 2.0
    return wd + we / 2.0 #return spin symmetrized pp ver4
end

function _measure_ver4(vars, obs, weights, config)
    varK, varT, varX, varN = vars
    ki = varN[1]
    obs[1][ki] += weights[1]
end

"""
    function vertex4_oneloop(para::ParaMC;

calculate particle-particle-pair * R. Only spin (up, up; down, down) is non-zero!

In the large kamp limit, with kamp2 =0, it approaches to -m*e^2*NF*pi/2/kamp
"""
function vertex4_oneloop(para::ParaMC;
    neval=1e6, #number of evaluations
    kamp=[para.kF,],
    kamp2=[para.kF for i in 1:length(kamp)],
    config=nothing,
    print=0,
    integrand=_ver4PP,
    filename=nothing,
    kwargs...
)

    UEG.MCinitialize!(para)
    dim, β, kF = para.dim, para.β, para.kF
    R = MCIntegration.Continuous(0.0, 1.0 - 1e-6) # a small cutoff to make sure R is not 1
    Theta = MCIntegration.Continuous(0.0, 1π)
    Phi = MCIntegration.Continuous(0.0, 2π)
    K = CompositeVar(R, Theta, Phi)
    T = MCIntegration.Continuous(0.0, β, offset=1)
    T.data[1] = 0.0
    X = MCIntegration.Continuous(-1.0, 1.0) #x=cos(θ)
    N = Discrete(1, length(kamp))

    dof = [[1, 3, 1, 1],]
    obs = [zeros(ComplexF64, length(kamp)),]

    if isnothing(config)
        config = MCIntegration.Configuration(
            var=(K, T, X, N),
            dof=dof,
            obs=obs,
            type=ComplexF64,
            userdata=(para, kamp, kamp2),
            kwargs...
        )
    end

    result = integrate(integrand;
        measure=_measure_ver4,
        config=config, neval=neval, print=print, solver=:vegasmc, kwargs...)

    if isnothing(result) == false

        avg, std = result.mean[1], result.stdev[1]
        r = measurement.(real(avg), real(std))
        i = measurement.(imag(avg), imag(std))
        ver4 = Complex.(r, i)

        if isnothing(filename) == false
            jldopen(filename, "a+") do f
                key = "$(UEG.short(para))"
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                f[key] = (kamp, ver4)
            end
        end

        return ver4, result
    end
end



