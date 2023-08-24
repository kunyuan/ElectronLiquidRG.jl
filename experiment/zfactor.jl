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

const rs = 5.0
const beta = 25.0
const mass2 = 0.001
const order = 1

const para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order + 1)

# const Fs = RG.fdict[para.rs]
# const Fs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0] #must be in increasing order
# const Fs = [-0.3, -0.2, -0.1, 0.0] #must be in increasing order
const Fs = [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0] #must be in increasing order
# const Λgrid = RG.Λgrid(para.kF)
const Λgrid = CompositeGrid.LogDensedGrid(:gauss, [1.0 * para.kF, 100 * para.kF], [para.kF,], 8, 0.1 * para.kF, 8)
const sparseΛgrid = RG.SparseΛgrid(para.kF)

println("Λgrid = ", Λgrid.grid / para.kF)

function get_z()

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order)

    dzi, dmu, dz = RG.zCT(_para, "data/sigma.jld2"; Fs=Fs, Λgrid=sparseΛgrid)

    z1 = dz[1]

    println(z1[:, 1])

    return dz
end

function get_ver3(dz, dz2)
    para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order)
    f_pp = jldopen("data/ver3_PP.jld2", "r")
    f_phe = jldopen("data/ver3_PHE.jld2", "r")
    # f_ph = jldopen("data/ver3_PH.jld2", "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))
    ver3_pp = MeshArray(Fs, sparseΛgrid; dtype=Complex{Measurement{Float64}})
    ver3_phe = MeshArray(Fs, sparseΛgrid; dtype=Complex{Measurement{Float64}})
    ver3_ph = MeshArray(Fs, sparseΛgrid; dtype=Complex{Measurement{Float64}})

    for (fi, F) in enumerate(Fs)
        # sometimes, MC.jl calculate the vertex3 with order = 2, the results are not affected
        _para1 = RG.get_para(para, F; order=1)
        _para2 = RG.get_para(para, F; order=2)
        key1 = UEG.short(_para1)
        key2 = UEG.short(_para2)
        if haskey(f_pp, key1)
            key = key1
        elseif haskey(f_pp, key2)
            key = key2
        else
            error("key not found")
        end
        kgrid, _ver3 = f_pp[key]
        ver3_pp[fi, :] = _ver3
        @assert kgrid ≈ sparseΛgrid

        kgrid, _ver3 = f_phe[key]
        ver3_phe[fi, :] = _ver3
        @assert kgrid ≈ sparseΛgrid

        # kgrid, _ver3 = f_ph[key]
        # ver3_ph[fi, :] = _ver3
        # @assert kgrid ≈ sparseΛgrid
        for (fi, Fs) in enumerate(ver3_ph.mesh[1])
            _para = RG.get_para(para, Fs)
            for (ki, k) in enumerate(ver3_ph.mesh[2])
                ver3_ph[fi, ki] = RG.Lbubble(_para, k, para.kF)
            end
        end
    end

    return ver3_pp, ver3_phe, ver3_ph
end

function get_ver4(dz, dz2)

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order + 1)

    vuu, vud = RG.vertex4_renormalize(_para, "data/ver4.jld2", dz, dz2; Fs=Fs, Λgrid=sparseΛgrid)

    return vuu, vud
end

# function print_ver3(ver3, fi=size(ver3)[1]; nsample=5)
function print_ver3(ver3_pp, ver3_phe, ver3_ph; fi=1, nsample=5)
    kF = para.kF
    Fs = ver3_pp.mesh[1]
    kgrid = ver3_pp.mesh[2]
    step = Int(ceil(length(kgrid) / nsample))
    kidx = [i for i in 1:step:length(kgrid)]
    # fi = length(Fs)
    println("Fs = $(Fs[fi]) at index $fi")
    printstyled(@sprintf("%12s    %24s    %24s    %24s\n",
            "k/kF", "ver3_pp", "ver3_phe", "ver3_ph"), color=:yellow)
    for ki in kidx
        @printf("%12.6f    %24s    %24s    %24s\n", kgrid[ki] / kF, real.(ver3_pp[fi, ki]) / para.NF, real.(ver3_phe[fi, ki]) / para.NF, real.(ver3_ph[fi, ki]) / para.NF)
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
                b_fine[fi, ki], b_deriv[fi, ki] = b_tail(finekgrid[ki])
            else
                # b_deriv[fi, ki] = db[ki]
                # b_fine[fi, ki] = smoothed[ki]

                b_fine[fi, ki] = interp_b(Fmesh[fi], finekgrid[ki], b)
            end
        end

        # it is important to derive the value on fine mesh first, then take derivative with spline fitting
        _w = [_d.err for _d in b_fine[fi, :]]
        _b_data = [_d.val for _d in b_fine[fi, :]]
        smoothed, db = compute_derivative(finekgrid.grid, _b_data, finekgrid, s=(maximum(_w))^2 * 2)
        b_deriv[fi, :] .= db

        # for ki in eachindex(finekgrid)
        #     b_deriv[fi, ki] = Interp.differentiate1D(b_fine[fi, :], finekgrid, finekgrid[ki])
        # end
    end

    for fi in eachindex(Fmesh)

        # for ki in 1:length(finekgrid)-1
        #     # println(finekgrid[ki] / para.kF, ", ", b_fine[fi, ki], ", ", b_deriv[fi, ki], ", ",
        #     # Interp.integrate1D(b_deriv[fi, :], finekgrid, [finekgrid[ki], finekgrid[ki+1]])
        #     # )
        #     temp = Interp.integrate1D(b_deriv[fi, :], finekgrid, [finekgrid[ki], finekgrid[ki+1]])
        #     target = b_fine[fi, ki+1] - b_fine[fi, ki]
        #     @printf("%12.6f    %12.6f    %12.6f    %12.6f   %12.6f\n", finekgrid[ki] / para.kF, b_fine[fi, ki], b_deriv[fi, ki], temp, target)
        # end

        test = [-Interp.integrate1D(b_deriv[fi, :], finekgrid, [kgrid[i], finekgrid[end]]) + b_tail(finekgrid[end])[1] for i in 1:length(kgrid)]
        # test = test ./ kgrid

        nsample = 5
        # step = Int(ceil(length(finekgrid) / nsample))
        # kidx = [i for i in 1:step:length(finekgrid)]
        kidx = collect(1:length(kgrid))

        println("Fs = ", Fmesh[fi])
        @printf("%12s    %12s    %12s    %12s    %12s\n", "k/kF", "b_deriv", "b_tail", "b_finegrid", "int b_deriv")
        for ki in kidx
            k = kgrid[ki]
            dkidx = searchsortedfirst(finekgrid, k)
            @printf("%12.6f    %12.6f    %12.6f    %12.6f    %12.6f\n", k / para.kF, b_deriv[fi, dkidx], b_tail(k)[2], b[fi, ki], test[ki])
            # println("$(k / para.kF)    $(b_deriv[fi, ki]) $(b_tail(k)[2])   $(b[fi, ki])     $(test[ki])")
        end
    end

    return b_fine, b_deriv
end

function c_on_fine_grid(a)

    Fmesh = SimpleG.Arbitrary{Float64}(a.mesh[1])
    kgrid = a.mesh[2]
    finekgrid = Λgrid
    c_deriv = MeshArray(a.mesh[1], finekgrid; dtype=Measurement{Float64})

    for ki in eachindex(finekgrid)
        if finekgrid[ki] >= kgrid[end]
            c_deriv[:, ki] .= para.me / 8 / π
        else
            c1 = RG.c_coeff_pp(para, finekgrid[ki] + para.kF * 0.01, para.kF)[2]
            c2 = RG.c_coeff_pp(para, finekgrid[ki], para.kF)[2]
            c_deriv[:, ki] .= (c1 - c2) / (0.01 * para.kF)
            # b2 = interp_b(Fmesh[fi], finekgrid[ki+1], b)
            # b_deriv[fi, ki] = (b1.val - b2.val) / (finekgrid[ki] - finekgrid[ki+1])
        end
    end

    println("c_deriv: ", c_deriv[1, 1], " -> ", c_deriv[1, end])

    return c_deriv / para.NF
end

function get_u(Fs, Λ)
    para_new = UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs, Fa=-0.0, isDynamic=true)
    return RG.u_from_f(para_new, Λ, para.kF)[1]
end

function solve_RG(a, b_deriv, c_deriv; max_iter=20, mix=0.5)

    function Rex(Fs, k)
        _p = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs, Fa=-0.0, isDynamic=true, order=1)
        return RG.R_exchange(_p, k, _p.kF)
    end

    Fmesh = SimpleG.Arbitrary{Float64}(a.mesh[1])
    Kmesh = a.mesh[2]

    interp(Fs::Float64, k::Float64, v) = RG.linear_interp(v, Fmesh, Kmesh, Fs, k)


    Λgrid = Kmesh

    # Fs_Λ = zeros(length(Λgrid))
    Fs_Λ = [RG.KO(para, l, para.kF)[1] for l in Λgrid]

    para_Λ = [UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ[ui], Fa=-0.0, isDynamic=true) for ui in eachindex(Λgrid)]
    # ∂R_∂Λ = [RG.∂R_∂Λ_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    # ∂R_∂F = [RG.∂R_∂F_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    # println(" ∂R_∂Λ : ", ∂R_∂Λ)
    # println(" ∂R_∂F : ", ∂R_∂F)
    # dFs_Λ = [∂R_∂Λ[i] / (1 - ∂R_∂F[i]) for i in eachindex(Λgrid)]
    # dFs_Λ = [∂R_∂Λ[i] / (1 - ∂R_∂F[i]) for i in eachindex(Λgrid)]

    # smoothed, dFs_Λ = compute_derivative(Λgrid, Fs_Λ, s=0.0)
    # plot_fit(Λgrid, Fs_Λ, smoothed, dFs_Λ)

    u_Λ = zeros(length(Λgrid))
    # du_Λ = zeros(length(Λgrid))

    # println("Step 1: solve for without b and c")

    for i in 1:max_iter
        Fs_Λ_new = zeros(length(Λgrid))
        u_Λ_new = zeros(length(Λgrid))

        _a = [interp(Fs_Λ[i], Λgrid[i], a).val for i in eachindex(Λgrid)]
        _b_deriv = [interp(Fs_Λ[i], Λgrid[i], b_deriv).val for i in eachindex(Λgrid)]
        _c_deriv = [interp(Fs_Λ[i], Λgrid[i], c_deriv).val for i in eachindex(Λgrid)]

        for ui in eachindex(Λgrid)
            k = Λgrid[ui]

            _u = u_Λ[ui]
            _Fs = Fs_Λ[ui]

            _c = Interp.integrate1D(_c_deriv .* u_Λ .^ 2, Λgrid, [Λgrid[ui], Λgrid[end]])
            _b = Interp.integrate1D(_b_deriv .* u_Λ, Λgrid, [Λgrid[ui], Λgrid[end]])

            Fs_Λ_new[ui] = -Rex(_Fs, k) + _a[ui] - _b - _c
            # Fs_Λ_new[ui] = -Rex(_Fs, k) + _a[ui] - _b
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

    # w = [a(Fs_Λ[ki], k, vs2).err for (ki, k) in enumerate(Λgrid)]
    # smoothed, dFs_Λ_new = compute_derivative(Λgrid, Fs_Λ, s=(maximum(w))^2 * 2)

    # # plot_fit(Λgrid, Fs_Λ, smoothed, dFs_Λ)
    # dFs_Λ .= dFs_Λ * mix .+ dFs_Λ_new * (1 - mix)
    # # exit(0)

    # para_Λ = [UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ[ui], Fa=-0.0, isDynamic=true) for ui in eachindex(Λgrid)]
    # ∂R_∂Λ = [RG.∂R_∂Λ_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    # ∂R_∂F = [RG.∂R_∂F_exchange(para_Λ[i], k, para.kF) for (i, k) in enumerate(Λgrid)]
    # du_Λ = [(1 + ∂R_∂F[i]) * dFs_Λ[i] + ∂R_∂Λ[i] for i in eachindex(Λgrid)]

    println(Fs_Λ[1], ", ", u_Λ[1])

    return u_Λ, Fs_Λ
end

# solve the RG equation as a differential equation
function solve_RG2(a, b_deriv, c_deriv; maxiter=100, mix=0.5)

    function Rex(Fs, k)
        _p = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs, Fa=-0.0, isDynamic=true, order=1)
        return RG.R_exchange(_p, k, _p.kF)
    end

    Fmesh = SimpleG.Arbitrary{Float64}(a.mesh[1])
    Kmesh = a.mesh[2]

    interp(Fs::Float64, k::Float64, v) = RG.linear_interp(v, Fmesh, Kmesh, Fs, k)

    Λgrid = deepcopy(Kmesh)

    Fs_Λ = [RG.KO(para, l, para.kF)[1] for l in Λgrid]
    u_Λ = zeros(length(Λgrid))

    a_Λ = zeros(length(Λgrid))
    db_Λ = zeros(length(Λgrid))
    dc_Λ = zeros(length(Λgrid))

    #from the last scale to the Fermi surface, inverse iteration
    for i in reverse(1:length(Λgrid)-1)
        Λ = Λgrid[i]

        Fs_Λ[i] = Fs_Λ[i+1]
        u_Λ[i] = get_u(Fs_Λ[i], Λ) #temporary value
        for iter in 1:maxiter
            Fs = Fs_Λ[i]

            a_Λ[i] = interp(Fs, Λ, a).val
            db_Λ[i] = interp(Fs, Λ, b_deriv).val
            # db_Λ[i] = interp(-0.2, Λ, b_deriv).val
            dc_Λ[i] = interp(Fs, Λ, c_deriv).val

            # println(db_Λ[i:end])
            _b = -Interp.integrate1D(db_Λ .* u_Λ, Λgrid, [Λ, Λgrid[end]])
            _c = -Interp.integrate1D(dc_Λ .* u_Λ .^ 2, Λgrid, [Λ, Λgrid[end]])

            Fs_Λ_new = -Rex(Fs, Λ) + a_Λ[i] + _b * 1.0 + _c * 1.0
            # u_Λ_new = get_u(Fs, Λ) #temporary value

            # u_Λ[i] = get_u(Fs_Λ[i], Λ)
            if abs(Fs_Λ_new - Fs_Λ[i]) < 1e-4
                break
            end

            if iter >= maxiter - 1
                println("Warning: max iteration reached: ", iter, " with diff ", abs(Fs_Λ_new - Fs_Λ[i]), " at scale ", Λ / para.kF, " with ", Fs_Λ_new[i])
            end

            println(iter, ": Fs = ", Fs_Λ_new, " and u =", u_Λ[i])

            Fs_Λ[i] = Fs_Λ[i] * mix + Fs_Λ_new * (1 - mix)
            u_Λ[i] = get_u(Fs_Λ[i], Λ)
        end

    end

    println("#k/kF         Fs                u")
    for i in eachindex(Λgrid)
        # println(Λgrid[i] / para.kF, "    ", Fs_Λ[i], "    ", u_Λ[i])
        @printf("%12.6f    %24s    %24s\n", Λgrid[i] / para.kF, Fs_Λ[i], u_Λ[i])
    end
    # println(Fs_Λ[1], ", ", u_Λ[1])

    return u_Λ, Fs_Λ
end

dz = get_z()
dz2 = [dz[1] for i in 1:length(dz)] # right leg is fixed to the Fermi surface 
ver3_pp, ver3_phe, ver3_ph = get_ver3(dz, dz2)
print_ver3(ver3_pp, ver3_phe, ver3_ph; nsample=5)

ver3 = ver3_pp + ver3_phe + ver3_ph

b = -real.(ver3) / 2.0 #ver3 only has up, up, down, down spin configuration, project to spin symmetric channel.
# the extra minus sign is because ver3 sign is defined as (-1)^order where order is the number of interaction lines, which is 2 for ver3. But here we need the sign to be (-1)^(order-1) = -1
b = b .* 2 .* 2 # uGGR + RGGu contributes factor of 2, then u definition contributes another factor of 2

b = b / para.NF

# b = b + dz[1]

b_fine, b_deriv = b_on_fine_grid(b)

c_deriv = c_on_fine_grid(b)

vuu, vud = get_ver4(dz, dz2)
print_ver4(vuu, vud, 1; nsample=5)
# print_ver4(vuu, vud, 2; nsample=10)
# print_ver4(vuu, vud, 3; nsample=10)
print_ver4(vuu, vud; nsample=5)

# vs, vs_deriv = a_derivative(vuu, vud)

vs = -real.((vuu + vud)) / 2.0 # -Gamma4 (multi-loop Feynman diagrams are for -Gamma4)
vs = vs .* 2.0 # u definition has a factor of 2
a_fine = a_on_fine_grid(vs[2])


_u, _Fs = solve_RG2(a_fine, b_deriv, c_deriv)
println(_Fs[1])
println(_u[1])
