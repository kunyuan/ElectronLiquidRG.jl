using ElectronLiquid
import ElectronLiquidRG as RG
using Measurements
using GreenFunc
using JLD2
using Printf
# using DifferentialEquations
using CompositeGrids
using PyCall
using Plots

const rs = 1.0
const beta = 25.0
const mass2 = 0.01
const order = 1

const para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order + 1)

# const Fs = RG.fdict[para.rs]
const Fs = [-0.3, -0.2, -0.1, 0.0] #must be in increasing order
# const Λgrid = RG.Λgrid(para.kF)
const Λgrid = CompositeGrid.LogDensedGrid(:gauss, [1.0 * para.kF, 100 * para.kF], [para.kF,], 8, 0.2 * para.kF, 4)
const sparseΛgrid = RG.SparseΛgrid(para.kF)

function get_z()

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order)

    dzi, dmu, dz = RG.zCT(_para, "data/sigma.jld2"; Fs=Fs, Λgrid=sparseΛgrid)

    z1 = dz[1]

    println(z1[:, 1])

    return dz
end

function get_ver3()
    para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order)
    f_pp = jldopen("data/ver3_PP.jld2", "r")
    f_phe = jldopen("data/ver3_PHE.jld2", "r")
    f_ph = jldopen("data/ver3_PH.jld2", "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))
    ver3_pp = MeshArray(Fs, sparseΛgrid; dtype=Complex{Measurement{Float64}})
    ver3_phe = MeshArray(Fs, sparseΛgrid; dtype=Complex{Measurement{Float64}})
    ver3_ph = MeshArray(Fs, sparseΛgrid; dtype=Complex{Measurement{Float64}})

    for (fi, F) in enumerate(Fs)
        _para = RG.get_para(para, F)
        key = UEG.short(_para)
        kgrid, _ver3 = f_pp[key]
        ver3_pp[fi, :] = _ver3
        @assert kgrid ≈ sparseΛgrid

        kgrid, _ver3 = f_phe[key]
        ver3_phe[fi, :] = _ver3
        @assert kgrid ≈ sparseΛgrid

        kgrid, _ver3 = f_ph[key]
        ver3_ph[fi, :] = _ver3
        @assert kgrid ≈ sparseΛgrid
    end

    return ver3_pp, ver3_phe, ver3_ph
end

function get_ver4(dz, dz2)

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order + 1)

    vuu, vud = RG.vertex4_renormalize(_para, "data/ver4.jld2", dz, dz2; Fs=Fs, Λgrid=sparseΛgrid)

    return vuu, vud
end

function print_ver3(ver3, fi=size(ver3)[1]; nsample=5)
    kF = para.kF
    Fs = ver3.mesh[1]
    kgrid = ver3.mesh[2]
    step = Int(ceil(length(kgrid) / nsample))
    kidx = [i for i in 1:step:length(kgrid)]
    # fi = length(Fs)
    println("Fs = $(Fs[fi]) at index $fi")
    printstyled(@sprintf("%12s    %24s\n",
            "k/kF", "ver3"), color=:yellow)
    for ki in kidx
        @printf("%12.6f    %24s\n", kgrid[ki] / kF, "$(ver3[fi, ki])")
    end
end


function print_ver4(vuu, vud, fi=size(vuu[1])[1]; nsample=5)

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order + 1)

    kF = para.kF
    # Fs = RG.fdict[para.rs]
    # kgrid= RG.Λgrid(para.kF)
    sample = vuu[1]
    Fs = sample.mesh[1]
    kgrid = sample.mesh[2]
    step = Int(ceil(length(kgrid) / nsample))
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
            d1, d2 = real(vuu[p][fi, ki]), real(vud[p][fi, ki])
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
        d1, d2 = real(_vuu[fi, ki]), real(_vud[fi, ki])
        # s, a = (d1 + d2) / 2.0, (d1 - d2) / 2.0
        s, a = Ver4.ud2sa(d1, d2)
        @printf("%12.6f    %24s    %24s    %24s    %24s\n", kgrid[ki] / kF, "$d1", "$d2", "$s", "$a")
    end
end

# py"""
# import numpy as np
# from scipy.interpolate import InterpolatedUnivariateSpline

# def compute_derivative(x, y):
#     spline = InterpolatedUnivariateSpline(x, y, k=2)  # k=2 for quadratic
#     y_prime = spline.derivative()(x)
#     return y_prime
# """

py"""
import numpy as np
from scipy.interpolate import UnivariateSpline

def compute_derivative(x, y, finegrid, s = None):
    # Fit a smoothing spline
    spline = UnivariateSpline(x, y, k=3, s=s)
    
    # Compute the derivative
    y_prime = spline.derivative()(finegrid)
    
    return spline(finegrid), y_prime
"""

# Create a Python function reference in Julia
compute_derivative = pyimport("__main__").compute_derivative

function plot_fit(Λgrid, finegrid, Fs_Λ, smoothed, dFs_Λ)
    p1 = plot(Λgrid ./ para.kF, Fs_Λ, xlims=[1.0, 10.0], seriestype=[:scatter, :path], linestyle=:dash, mark=:dot, label="Noisy Data", title="Noisy Data")
    plot!(p1, finegrid ./ para.kF, smoothed, label="smoothed Data", title="smoothed Data")
    p2 = plot(finegrid ./ para.kF, dFs_Λ, xlims=[1.0, 10.0], seriestype=[:scatter, :path], linestyle=:dash, mark=:dot, label="Derived Data", title="Derivative", color=:red)
    p = plot(p1, p2, layout=(1, 2), size=(800, 400))
    display(p)
    readline()
end

function b_tail(k)
    factor = para.NF / 2 * 4
    B = -π * para.me * para.e0^2 / 2 * factor
    return B / k, -B / k^2
end

function a_on_fine_grid(a)
    Fmesh = SimpleG.Arbitrary{Float64}(a.mesh[1])
    kgrid = a.mesh[2]
    finekgrid = Λgrid

    interp_a(Fs::Float64, k::Float64, ver) = RG.linear_interp(ver, Fmesh, kgrid, Fs, k)

    a_fine = MeshArray(b.mesh[1], finekgrid; dtype=Measurement{Float64})
    for fi in eachindex(Fmesh)
        # take derivation of kgrid
        for ki in eachindex(finekgrid)
            if finekgrid[ki] >= kgrid[end]
                a_fine[fi, ki] = 0.0
            else
                a_fine[fi, ki] = interp_a(Fmesh[fi], finekgrid[ki], a)
                # b2 = interp_b(Fmesh[fi], finekgrid[ki+1], b)
                # b_deriv[fi, ki] = (b1.val - b2.val) / (finekgrid[ki] - finekgrid[ki+1])
            end
        end
    end
    return a_fine
end

function b_on_fine_grid(b)

    Fmesh = SimpleG.Arbitrary{Float64}(b.mesh[1])
    kgrid = b.mesh[2]
    finekgrid = Λgrid
    # finekgrid = kgrid
    # finekgrid = CompositeGrid.LogDensedGrid(:gauss, [1.0 * para.kF, 100 * para.kF], [para.kF,], 8, 0.2 * para.kF, 8)

    interp_b(Fs::Float64, k::Float64, ver3) = RG.linear_interp(ver3, Fmesh, kgrid, Fs, k)


    # b_deriv = similar(b)
    b_fine = MeshArray(b.mesh[1], finekgrid; dtype=Measurement{Float64})
    b_deriv = MeshArray(b.mesh[1], finekgrid; dtype=Measurement{Float64})
    for fi in eachindex(Fmesh)
        bΛ = b[fi, :]
        w = [bΛ[ki].err for (ki, k) in enumerate(kgrid)]
        data = [d.val for d in bΛ]
        smoothed, db = compute_derivative(kgrid.grid, data, finekgrid, s=(maximum(w))^2 * 2)
        # plot_fit(kgrid, finekgrid, bΛ, smoothed, b_deriv[fi, :])

        # take derivation of kgrid
        for ki in eachindex(finekgrid)
            if ki >= length(finekgrid) || finekgrid[ki] >= kgrid[end]
                b_fine[fi, ki], b_deriv[fi, ki] = b_tail(finekgrid[ki])[2]
            else
                b_deriv[fi, ki] = db[ki]
                b_fine[fi, ki] = smoothed[ki]
                # b1 = interp_b(Fmesh[fi], finekgrid[ki], b)
                # b2 = interp_b(Fmesh[fi], finekgrid[ki+1], b)
                # b_deriv[fi, ki] = (b1.val - b2.val) / (finekgrid[ki] - finekgrid[ki+1])
            end
        end
    end

    for fi in eachindex(Fmesh)
        test = [-Interp.integrate1D(b_deriv[fi, :], finekgrid, [kgrid[i], finekgrid[end]]) + b_tail(finekgrid[end])[1] for i in 1:length(kgrid)]
        # test = test ./ kgrid

        println("k/kF     dvs     dvs_fitted     test: ", Fmesh[fi])
        for (ki, k) in enumerate(kgrid)
            println("$(k / para.kF)    $(b_deriv[fi, ki]) $(b_tail(k)[2])   $(b[fi, ki])     $(test[ki])")
        end
    end

    return b_fine, b_deriv
end

function solve_RG(vuu, vud, ver3; max_iter=20, mix=0.5)
    function Rex(Fs, k)
        _p = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs, Fa=-0.0, isDynamic=true, order=1)
        return RG.R_exchange(_p, k, _p.kF)
    end

    Fmesh = SimpleG.Arbitrary{Float64}(vuu[1].mesh[1])
    Kmesh = vuu[1].mesh[2]

    a(Fs::Float64, k::Float64, vs) = RG.linear_interp(vs, Fmesh, Kmesh, Fs, k) * 2
    b(Fs::Float64, k::Float64, ver3) = RG.linear_interp(ver3, Fmesh, Kmesh, Fs, k) / para.NF * 4

    vs = -real.((vuu + vud)) / 2.0 # -Gamma4 (multi-loop Feynman diagrams are for -Gamma4)
    vs1, vs2 = vs[1], vs[2]
    ver3 = -real.(ver3) / 2.0 # -Gamma3 (because the direct interaction is negative)
    # println(" vs : ", vs)
    # println(" ver3 : ", ver3)
    println(size(vs2))
    println(Fmesh)
    for i in eachindex(Fmesh)
        println(vs2.mesh[1][i], " -> ", vs2[i, 1])
    end
    println(a(-0.1, para.kF, vs2))

    c = para.me / 8 / π / para.NF # ~0.2 for rs=1

    # Λgrid = RG.Λgrid(para.kF)

    # Fs_Λ = zeros(length(Λgrid))
    Fs_Λ = [RG.KO(para, l, para.kF)[1] for l in Λgrid]

    para_Λ = [UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ[ui], Fa=-0.0, isDynamic=true) for ui in eachindex(Λgrid)]
    # ∂R_∂Λ = [RG.∂R_∂Λ_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    # ∂R_∂F = [RG.∂R_∂F_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    # println(" ∂R_∂Λ : ", ∂R_∂Λ)
    # println(" ∂R_∂F : ", ∂R_∂F)
    # dFs_Λ = [∂R_∂Λ[i] / (1 - ∂R_∂F[i]) for i in eachindex(Λgrid)]
    # dFs_Λ = [∂R_∂Λ[i] / (1 - ∂R_∂F[i]) for i in eachindex(Λgrid)]

    smoothed, dFs_Λ = compute_derivative(Λgrid, Fs_Λ, s=0.0)
    plot_fit(Λgrid, Fs_Λ, smoothed, dFs_Λ)

    u_Λ = zeros(length(Λgrid))
    du_Λ = zeros(length(Λgrid))

    println("Step 1: solve for without b and c")

    for i in 1:max_iter
        Fs_Λ_new = zeros(length(Λgrid))
        u_Λ_new = zeros(length(Λgrid))

        for ui in eachindex(Λgrid)
            k = Λgrid[ui]

            _u = u_Λ[ui]
            _Fs = Fs_Λ[ui]
            _a = a(_Fs, k, vs2).val

            Fs_Λ_new[ui] = -Rex(_Fs, k) + _a
            # Fs_Λ_new[ui] = -Rex(_Fs, k)
            # Fs_Λ_new[ui] = -Rex(_Fs, k) + _a - _b * u_Λ[ui] - _b_deriv + _c
            # Fs_Λ_new[ui] = -Rex(_Fs, k) + _a

            para_new = UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ_new[ui], Fa=-0.0, isDynamic=true)
            u_Λ_new[ui] = RG.u_from_f(para_new, k, para.kF)[1]
        end
        diff_u = maximum(abs.(u_Λ_new - u_Λ))
        diff_Fs = maximum(abs.(Fs_Λ_new - Fs_Λ))
        println("iter $i, diff_u = $diff_u, diff_Fs = $diff_Fs")
        # for li in eachindex(Λgrid)
        #     # println(Λgrid[li]/para.kF, " : ", Fs_Λ[li], " -> ", Fs_Λ_new[li], " = diff ", Fs_Λ_new[li]-Fs_Λ[li], " with ", vs2[end, li])
        #     println(Λgrid[li] / para.kF, " : ", Fs_Λ[li], " -> ", Fs_Λ_new[li], " = diff ", Fs_Λ_new[li] - Fs_Λ[li], " with ", a(Fs_Λ_new[li], Λgrid[li], vs2))
        # end
        # println("before: ", Fs_Λ)
        # println("after: ", Fs_Λ_new)
        # println("diff: ", Fs_Λ_new-Fs_Λ)
        if diff_Fs / maximum(abs.(Fs_Λ)) < 1e-4
            break
        end

        println(Fs_Λ_new[1], ", ", u_Λ_new[1])
        # exit(0)

        u_Λ .= u_Λ * mix .+ u_Λ_new * (1 - mix)
        Fs_Λ .= Fs_Λ * mix .+ Fs_Λ_new * (1 - mix)

    end

    w = [a(Fs_Λ[ki], k, vs2).err for (ki, k) in enumerate(Λgrid)]
    smoothed, dFs_Λ_new = compute_derivative(Λgrid, Fs_Λ, s=(maximum(w))^2 * 2)

    # plot_fit(Λgrid, Fs_Λ, smoothed, dFs_Λ)
    dFs_Λ .= dFs_Λ * mix .+ dFs_Λ_new * (1 - mix)
    # exit(0)

    para_Λ = [UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ[ui], Fa=-0.0, isDynamic=true) for ui in eachindex(Λgrid)]
    ∂R_∂Λ = [RG.∂R_∂Λ_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    ∂R_∂F = [RG.∂R_∂F_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    du_Λ = [(1 + ∂R_∂F[i]) * dFs_Λ[i] + ∂R_∂Λ[i] for i in eachindex(Λgrid)]

    println(Fs_Λ[1], ", ", u_Λ[1])

    # exit(0)
    println("Step 2: solve for u(Λ)")

    for i in 1:max_iter
        Fs_Λ_new = zeros(length(Λgrid))
        u_Λ_new = zeros(length(Λgrid))
        dFs_Λ_new = zeros(length(Λgrid))

        b_Λ = [b(Fs_Λ[ki], k, ver3).val for (ki, k) in enumerate(Λgrid)]
        println("b: ", b_Λ[1])
        for ui in eachindex(Λgrid)
            k = Λgrid[ui]

            _u = u_Λ[ui]
            _Fs = Fs_Λ[ui]
            _a = a(_Fs, k, vs2).val
            _b = b_Λ[ui]
            _c = c * Interp.integrate1D(u_Λ .^ 2, Λgrid, [Λgrid[ui], Λgrid[end]])
            _b_deriv = Interp.integrate1D(du_Λ .* b_Λ, Λgrid, [Λgrid[ui], Λgrid[end]])
            # println("_a = $_a, _b = $_b, _c = $_c, _b_deriv = $_b_deriv")

            # Fs_Λ_new[ui] = -Rex(_Fs, k) + _a + _b * u_Λ[ui] + _b_deriv - _c
            # Fs_Λ_new[ui] = -Rex(_Fs, k) + _a + _b * u_Λ[ui] + _b_deriv
            Fs_Λ_new[ui] = -Rex(_Fs, k) + _a + _b * u_Λ[ui]
            # Fs_Λ_new[ui] = -Rex(_Fs, k)
            # Fs_Λ_new[ui] = -Rex(_Fs, k) + _a - _b * u_Λ[ui] - _b_deriv + _c
            # Fs_Λ_new[ui] = -Rex(_Fs, k) + _a

            para_new = UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ_new[ui], Fa=-0.0, isDynamic=true)
            u_Λ_new[ui] = RG.u_from_f(para_new, k, para.kF)[1]
        end
        diff_u = maximum(abs.(u_Λ_new - u_Λ))
        diff_Fs = maximum(abs.(Fs_Λ_new - Fs_Λ))
        println("iter $i, diff_u = $diff_u, diff_Fs = $diff_Fs")
        # for li in eachindex(Λgrid)
        #     # println(Λgrid[li]/para.kF, " : ", Fs_Λ[li], " -> ", Fs_Λ_new[li], " = diff ", Fs_Λ_new[li]-Fs_Λ[li], " with ", vs2[end, li])
        #     println(Λgrid[li] / para.kF, " : ", Fs_Λ[li], " -> ", Fs_Λ_new[li], " = diff ", Fs_Λ_new[li] - Fs_Λ[li], " with ", a(Fs_Λ_new[li], Λgrid[li], vs2))
        # end
        # println("before: ", Fs_Λ)
        # println("after: ", Fs_Λ_new)
        # println("diff: ", Fs_Λ_new-Fs_Λ)
        if diff_Fs / maximum(abs.(Fs_Λ)) < 1e-4
            break
        end
        u_Λ .= u_Λ * mix .+ u_Λ_new * (1 - mix)
        Fs_Λ .= Fs_Λ * mix .+ Fs_Λ_new * (1 - mix)

        w = [a(Fs_Λ[ki], k, vs2).err for (ki, k) in enumerate(Λgrid)]
        smoothed, dFs_Λ_new = compute_derivative(Λgrid, Fs_Λ, s=(maximum(w))^2 * 2)

        # plot_fit(Λgrid, Fs_Λ, smoothed, dFs_Λ_new)
        # dFs_Λ .= dFs_Λ * mix .+ dFs_Λ_new * (1 - mix)
        # exit(0)

        # para_Λ = [UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ[ui], Fa=-0.0, isDynamic=true) for ui in eachindex(Λgrid)]
        # ∂R_∂Λ = [RG.∂R_∂Λ_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
        # ∂R_∂F = [RG.∂R_∂F_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
        # du_Λ = [(1 + ∂R_∂F[i]) * dFs_Λ[i] + ∂R_∂Λ[i] for i in eachindex(Λgrid)]
    end


    u_ref = [-Interp.integrate1D(du_Λ, Λgrid, [Λgrid[i], Λgrid[end]]) for i in eachindex(Λgrid)]
    for li in eachindex(Λgrid)
        println(Λgrid[li] / para.kF, " : ", Fs_Λ[li], " - ", u_Λ[li], " ref ", u_ref[li])
    end

    # du_Λ = compute_derivative(Λgrid, u_Λ)

    # Plot the original data and the derivative
    # p1 = plot(Λgrid, u_Λ, label="Noisy Data", title="Noisy Data")
    # p2 = plot(Λgrid, du_Λ, label="Derived Data", title="Derivative", color=:red)

    @assert all(diff(Λgrid) .> 0)

    w = [a(Fs_Λ[ki], k, vs2).err for (ki, k) in enumerate(Λgrid)]
    println(maximum(w))
    # println(length(w), ", ", length(Λgrid))
    smoothed, dFs_Λ = compute_derivative(Λgrid, Fs_Λ, s=(maximum(w))^2 * 2)
    plot_fit(Λgrid, Fs_Λ, smoothed, dFs_Λ)
    # p1 = plot(Λgrid ./ para.kF, Fs_Λ, xlims=[1.0, 5.0], seriestype=[:scatter, :path], linestyle=:dash, mark=:dot, label="Noisy Data", title="Noisy Data")
    # plot!(p1, Λgrid ./ para.kF, smoothed, label="smoothed Data", title="smoothed Data")
    # p2 = plot(Λgrid ./ para.kF, dFs_Λ, xlims=[1.0, 5.0], seriestype=[:scatter, :path], linestyle=:dash, mark=:dot, label="Derived Data", title="Derivative", color=:red)
    # p = plot(p1, p2, layout=(1, 2), size=(800, 400))
    # display(p)
    # readline()
    return u_Λ, Fs_Λ, du_Λ
end

dz = get_z()
dz2 = [dz[1] for i in 1:length(dz)] # right leg is fixed to the Fermi surface 
ver3_pp, ver3_phe, ver3_ph = get_ver3()
println("PP channel")
print_ver3(ver3_pp; nsample=10)
println("PHE channel")
print_ver3(ver3_phe; nsample=10)
println("PH channel")
print_ver3(ver3_ph; nsample=10)

ver3 = ver3_pp + ver3_phe + ver3_ph

b = -real.(ver3) / 2.0 #ver3 only has up, up, down, down spin configuration, project to spin symmetric channel
b = b .* 2 .* 2 # uGGR + RGGu contributes factor of 2, then u definition contributes another factor of 2

b_fine, b_deriv = b_on_fine_grid(b)

vuu, vud = get_ver4(dz, dz2)
print_ver4(vuu, vud, 1; nsample=10)
print_ver4(vuu, vud, 2; nsample=10)
print_ver4(vuu, vud, 3; nsample=10)
print_ver4(vuu, vud; nsample=10)

# vs, vs_deriv = a_derivative(vuu, vud)


_u, _Fs, _du = solve_RG(vuu, vud, ver3)
println(_Fs[1])
println(_u[1])
