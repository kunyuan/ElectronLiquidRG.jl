#Particle-particle diagram, order = 2
# sign = (-1)^order = 1
#
# kamp,up   kamp2,down
#   |--- u ---|
#   |         |
# up^         ^down
#   |         |
#   |-- KO ---|
#kamp,up  kamp2,down
# with Yukawa interaction mass2 = 0.1, we use Ver4.MC_PH to calculate the PPr diagram, where the leading order diagram should be the same as the particle-particle diagram, we obtain 
# order                upup            updown
#   1                 0.328(4)          0.0
#   2                 0.465(16)         0.661(22)
# therefore, we expect PP to be positively defined
function _PP(vars, config)
    varK, varT, varX, varN = vars
    R, Theta, Phi = varK
    para, kamp, kamp2 = config.userdata

    t1, t2 = varT[2], varT[3]
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

    qd = sqrt(dot(q, q))
    vq = UEG.interactionStatic(para, qd, 0.0, t1)
    wq = UEG.interactionDynamic(para, qd, 0.0, t1)

    # vq0 = -4π *para.e0^2 / (qd^2+para.mass2) /para.β 
    # @assert vq ≈ vq0 "vq=$vq, vq0=$vq0, qd=$qd, t1=$t1"
    # wq = 0.0
    # println(t1)

    phase_s = phase((0.0, t2, 0.0, t2), -1, 0, 0, para.β)
    phase_d = phase((0.0, t2, t1, t2), -1, 0, 0, para.β)

    factor = r^2 * sin(θ) / (1 - R[1])^2 / (2π)^3 * para.NF
    factor /= 2 # angle average with l=0

    # wud = g1 * (g2 * vq + g3 * wq) * factor * phase((0.0, t2, t1, t2), -1, 0, 0, para.β)
    wud = g1 * (g2 * vq * phase_s + g3 * wq * phase_d) * factor
    return wud * (+1)
end

# Left vertex correction with two external legs exchanged. Only (up, up; down, down) spin configuration will contribute. order = 2
# sign = (-1)^order * (-1 external leg exchange) * (-1 u exchange)  = 1
#
#   kamp,up   kamp2,down
#         \    /
#            x 
#         /    \
#       /        \
#       |-- < -\   \
#       |  down \    \
#       KO      |- u -|
#       |    up /     | 
#       |-- > -/      |
#       |             |
#   kamp,up       kamp2,down
function _Lver3(vars, config)
    varK, varT, varX, varN = vars
    R, Theta, Phi = varK
    para, kamp, kamp2 = config.userdata

    t1, t2 = varT[2], varT[3]
    r = R[1] / (1 - R[1])
    θ, ϕ = Theta[1], Phi[1]
    x = varX[1]
    ki = varN[1]
    kl, kr = kamp[ki], kamp2[ki]

    q = [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

    k1 = [kl + q[1], q[2], q[3]]
    k2 = [kr * x + q[1], kr * sqrt(1 - x^2) + q[2], +q[3]]

    ek1 = (dot(k1, k1) - para.kF^2) / 2 / para.me
    ek2 = (dot(k2, k2) - para.kF^2) / 2 / para.me

    g1 = Spectral.kernelFermiT(t2, ek1, para.β)
    g2 = Spectral.kernelFermiT(-t2, ek2, para.β)
    g3 = Spectral.kernelFermiT(t1 - t2, ek2, para.β)

    qd = sqrt(dot(q, q))
    vq = UEG.interactionStatic(para, qd, 0.0, t1)
    wq = UEG.interactionDynamic(para, qd, 0.0, t1)

    # vq0 = -4π *para.e0^2 / (qd^2+para.mass2) /para.β 
    # @assert vq ≈ vq0 "vq=$vq, vq0=$vq0, qd=$qd, t1=$t1"
    # wq = 0.0
    # println(t1)

    phase_s = phase((0.0, t2, t2, 0.0), -1, 0, 0, para.β)
    phase_d = phase((0.0, t2, t2, t1), -1, 0, 0, para.β)

    factor = r^2 * sin(θ) / (1 - R[1])^2 / (2π)^3 * para.NF
    factor /= 2 # angle average with l=0

    # wud = g1 * (g2 * vq + g3 * wq) * factor * phase((0.0, t2, t1, t2), -1, 0, 0, para.β)
    wud = g1 * (g2 * vq * phase_s + g3 * wq * phase_d) * factor
    return wud * (+1)
end

# Left vertex correction with two external legs exchanged. Only (up, up; down, down) spin configuration will contribute. order = 2
# sign = (-1)^order = 1
#
#   kamp,up       kamp2,down
#       |             |
#       |-- < -\      |
#       |  down \     |
#       KO      |- u -|
#       |  down /     | 
#       |-- > -/      |
#       |             |
#   kamp,up       kamp2,down
# with RPA interaction mass2 = 0.1, we use Ver4.MC_PH to calculate the PHr diagram, we obtain: 
# filter = [NoBubble,]
# order                upup            updown
#   1                 -24.28(21)        -24.60(22)
#   2                 -5.308(41)         -5.408(41)

# filter = [NoBubble, Proper]
# order                upup            updown
#   1                 -0.33056(98)       0.0
#   2                 -0.10125(76)       0.00

# _Lver3_direct = -0.010976 (56) = -5.517 / (v_q)/2
# therefore, we expect Lver3_direct to be negative
function _Lver3_direct(vars, config)
    varK, varT, varX, varN = vars
    R, Theta, Phi = varK
    para, kamp, kamp2 = config.userdata

    t1, t2 = varT[2], varT[3]
    r = R[1] / (1 - R[1])
    θ, ϕ = Theta[1], Phi[1]
    x = varX[1]
    ki = varN[1]
    kl, kr = kamp[ki], kamp2[ki]

    q = [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

    k1 = [kl + q[1], q[2], q[3]]
    # k2 = [kr * x + q[1], kr * sqrt(1 - x^2) + q[2], +q[3]]
    k2 = k1

    ek1 = (dot(k1, k1) - para.kF^2) / 2 / para.me
    ek2 = (dot(k2, k2) - para.kF^2) / 2 / para.me

    g1 = Spectral.kernelFermiT(t2, ek1, para.β)
    g2 = Spectral.kernelFermiT(-t2, ek2, para.β)
    g3 = Spectral.kernelFermiT(t1 - t2, ek2, para.β)

    qd = sqrt(dot(q, q))
    vq = UEG.interactionStatic(para, qd, 0.0, t1)
    wq = UEG.interactionDynamic(para, qd, 0.0, t1)

    # vq0 = -4π *para.e0^2 / (qd^2+para.mass2) /para.β 
    # @assert vq ≈ vq0 "vq=$vq, vq0=$vq0, qd=$qd, t1=$t1"
    # wq = 0.0
    # println(t1)

    phase_s = phase((0.0, 0.0, t2, t2), -1, 0, 0, para.β)
    phase_d = phase((0.0, t1, t2, t2), -1, 0, 0, para.β)

    factor = r^2 * sin(θ) / (1 - R[1])^2 / (2π)^3 * para.NF
    factor /= 2 # angle average with l=0

    # wud = g1 * (g2 * vq + g3 * wq) * factor * phase((0.0, t2, t1, t2), -1, 0, 0, para.β)
    wud = g1 * (g2 * vq * phase_s + g3 * wq * phase_d) * factor
    return wud * (+1)
end

# Left bubble correction with two external legs exchanged. Only (up, up; up, up) spin configuration will contribute.
# sign = (-1)^order * (-1 external leg exchange) * (-1 fermionic bubble) = 1
#
#   kamp,up    kamp2,up
#         \     /
#            x 
#         /     \
#       /         \
#     /   /- < -\   \
#   /    /  down \    \
#  |-KO-|        |- u -|
#  |     \  down /     | 
#  |      \- > -/      |
# kamp,up         kamp2,up
function _Lbubble(vars, config)
    varK, varT, varX, varN = vars
    R, Theta, Phi = varK
    para, kamp, kamp2 = config.userdata

    t1, t2 = varT[2], varT[3]
    r = R[1] / (1 - R[1])
    θ, ϕ = Theta[1], Phi[1]
    x = varX[1]
    ki = varN[1]
    kl, kr = kamp[ki], kamp2[ki]

    q = [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

    k1 = [kl + q[1], q[2], q[3]]
    k2 = [kr * x + q[1], kr * sqrt(1 - x^2) + q[2], +q[3]]

    ek1 = (dot(k1, k1) - para.kF^2) / 2 / para.me
    ek2 = (dot(k2, k2) - para.kF^2) / 2 / para.me

    g1 = Spectral.kernelFermiT(t2, ek1, para.β)
    g2 = Spectral.kernelFermiT(-t2, ek2, para.β)
    g3 = Spectral.kernelFermiT(t2 - t1, ek1, para.β)
    g4 = Spectral.kernelFermiT(t1 - t2, ek2, para.β)

    qt = k1 - k2
    qd = sqrt(dot(qt, qt))
    vq = UEG.interactionStatic(para, qd, 0.0, t1)
    wq = UEG.interactionDynamic(para, qd, 0.0, t1)

    # vq = -4π * para.e0^2 / (qd^2 + para.mass2) / para.β
    # @assert vq ≈ vq0 "vq=$vq, vq0=$vq0, qd=$qd, t1=$t1"
    # wq = 0.0
    # println(t1)

    _phase = phase((0.0, t2, t2, 0.0), -1, 0, 0, para.β)

    factor = r^2 * sin(θ) / (1 - R[1])^2 / (2π)^3 * para.NF
    factor /= 2 # angle average with l=0

    wuu = (g1 * g2 * vq + g3 * g4 * wq) * _phase * factor
    return wuu
end

@inline function phase(extT, ninL, noutL, ninR, β)
    # println(extT)
    tInL, tOutL, tInR, tOutR = extT
    winL, woutL, winR = π * (2ninL + 1) / β, π * (2noutL + 1) / β, π * (2ninR + 1) / β
    woutR = winL + winR - woutL
    return exp(-1im * (tInL * winL - tOutL * woutL + tInR * winR - tOutR * woutR))
end

function _measure_ver3(vars, obs, weights, config)
    varK, varT, varX, varN = vars
    ki = varN[1]
    obs[1][ki] += weights[1]
end

"""
    function vertex3(para::ParaMC;

calculate particle-particle-pair * R. Only spin (up, up; down, down) is non-zero!

In the large kamp limit, with kamp2 =0, it approaches to -m*e^2*NF*pi/2/kamp
"""
function vertex3(para::ParaMC;
    neval=1e6, #number of evaluations
    kamp=[para.kF,],
    kamp2=[para.kF for i in 1:length(kamp)],
    config=nothing,
    print=0,
    integrand=_PP,
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

    dof = [[1, 2, 1, 1],]
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
        measure=_measure_ver3,
        config=config, neval=neval, print=print, solver=:vegasmc, kwargs...)

    if isnothing(result) == false

        avg, std = result.mean[1], result.stdev[1]
        r = measurement.(real(avg), real(std))
        i = measurement.(imag(avg), imag(std))
        ver3 = Complex.(r, i)

        if isnothing(filename) == false
            jldopen(filename, "a+") do f
                key = "$(UEG.short(para))"
                if haskey(f, key)
                    @warn("replacing existing data for $key")
                    delete!(f, key)
                end
                f[key] = (kamp, ver3)
            end
        end

        return ver3, result
    end
end

# if abspath(PROGRAM_FILE) == @__FILE__
#     rs = [5.0,]
#     mass2 = [0.001,]
#     _Fs = [-2.5,]
#     beta = [25.0,]
#     order = [1,]
#     neval = 1e6

#     for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, _Fs, beta, order)
#         para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=1, mass2=_mass2, isDynamic=true, isFock=false)
#         result = vertex3(para;
#             neval=1e6,
#             integrand=_PP,
#             print=0
#         )
#         println(result)
#     end

# end

# Left bubble correction with two external legs exchanged. Only (up, up; up, up) spin configuration will contribute. order = 2
# sign = (-1)^order * (-1 external leg exchange) * (-1 fermionic bubble) = 1
#
#   kamp,up    kamp2,up
#         \     /
#            x 
#         /     \
#       /         \
#     /   /- < -\   \
#   /    /  down \    \
#  |-KO-|        |- u -|
#  |     \  down /     | 
#  |      \- > -/      |
# kamp,up         kamp2,up
# with Yukawa interaction mass2 = 0.01, we use Ver4.MC_PH to calculate the PHEr diagram, with filter=[], where the leading order diagram should be the same as the left bubble, we obtain 
# order                upup            updown
#   1                 1.261(15)          0.0
#   2                 -68.6(2.6)         -22.1(1.2)
# therefore, we expect Lbubble to be negative
function Lbubble(para::ParaMC, kamp=para.kF, kamp2=para.kF; ct=true,
    θgrid=CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 0.001, 32),
    kwargs...)
    kF = para.kF
    # println(kamp, ", ", kamp2)
    # qs = [2 * kamp * sin(θ / 2) for θ in θgrid.grid]
    qs = [sqrt(kamp^2 + kamp2^2 - 2 * cos(θ) * kamp * kamp2) for θ in θgrid.grid]
    # println(qs)

    Wp = zeros(Float64, length(qs))
    Wm = zeros(Float64, length(qs))
    for (qi, q) in enumerate(qs)
        Wp[qi] = UEG.KOstatic(q, para; ct=ct) * ElectronGas.Polarization.Polarization0_3dZeroTemp(q, 0, para.basic)
    end
    # Wp *= -NF * z^2 # additional minus sign because the interaction is exchanged
    # Wm *= -NF * z^2
    Wp *= para.NFstar  # additional minus sign because the interaction is exchanged
    Wm *= para.NFstar

    return Ver4.Legrendre(0, Wp, θgrid)
end
