using ElectronLiquid
import ElectronLiquidRG as RG
using Measurements
using GreenFunc
using JLD2
using Printf
# using DifferentialEquations
using CompositeGrids

const rs = 1.0
const beta = 25.0
const mass2 = 0.01
const order = 1

const para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order+1)

# const Fs = RG.fdict[para.rs]
const Fs = [0.0, -0.1, -0.2, -0.3]
const Λgrid= RG.Λgrid(para.kF)
const sparseΛgrid= RG.SparseΛgrid(para.kF)

function get_z()

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order)

    dzi, dmu, dz = RG.zCT(_para, "data/sigma.jld2"; Fs=Fs, Λgrid=sparseΛgrid)

    z1 = dz[1]

    println(z1[:, 1])

    return dz
end

function get_ver3()
    para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order+1)
    f = jldopen("data/ver3.jld2", "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))
    ver3 = MeshArray(Fs, sparseΛgrid; dtype=Complex{Measurement{Float64}})

    for (fi, F) in enumerate(Fs)
        _para = RG.get_para(para, F)
        key = UEG.short(_para)
        kgrid, _ver3 = f[key]
        ver3[fi, :] = _ver3
        @assert kgrid ≈ sparseΛgrid
    end
    return ver3
end

function get_ver4(dz)

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order+1)

    vuu, vud = RG.vertex4_renormalize(_para, "data/ver4.jld2", dz; Fs=Fs, Λgrid=sparseΛgrid)

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

    _para = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=-0.0, Fa=-0.0, isDynamic=true, order=order+1)

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

function solve_RG(vuu, vud, ver3; max_iter = 20, mix=0.5)
    function Rex(Fs, k)
        _p = ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs, Fa=-0.0, isDynamic=true, order=1)
        return RG.R_exchange(_p, k, _p.kF)
    end

    Fmesh = SimpleG.Arbitrary{Float64}(vuu[1].mesh[1])
    Kmesh = vuu[1].mesh[2]

    a(Fs::Float64, k::Float64, vs) = RG.linear_interp(vs, Fmesh, Kmesh, Fs, k)
    b(Fs::Float64, k::Float64, ver3) = RG.linear_interp(ver3, Fmesh, Kmesh, Fs, k)/para.NF 

    vs = -real.((vuu+vud))/2.0 # -Gamma4 (multi-loop Feynman diagrams are for -Gamma4)
    vs1, vs2 = vs[1], vs[2]
    ver3 = -real.(ver3) # -Gamma3 (because the direct interaction is negative)
    # println(" vs : ", vs)
    # println(" ver3 : ", ver3)
    println(vs2[end, 1])
    println(a(-0.1, para.kF, vs2))

    c = para.me/8/π/para.NF # ~0.2 for rs=1

    Λgrid = RG.Λgrid(para.kF)

    # Fs_Λ = zeros(length(Λgrid))
    Fs_Λ = [RG.KO(para, l, para.kF)[1] for l in Λgrid]
    u_Λ = zeros(length(Λgrid))
    du_Λ = zeros(length(Λgrid))

    for i in 1:max_iter
        Fs_Λ_new = zeros(length(Λgrid))
        u_Λ_new = zeros(length(Λgrid))
        para_Λ = [UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ[ui], Fa=-0.0, isDynamic=true) for ui in eachindex(Λgrid)]
        b_Λ = [b(Fs_Λ[ki], k, ver3).val for (ki, k) in enumerate(Λgrid)]
        for ui in eachindex(Λgrid)
            k = Λgrid[ui]

            _u = u_Λ[ui]
            _Fs = Fs_Λ[ui]
            _a = a(_Fs, k, vs2).val
            _b = b_Λ[ui] 
            _c = c*Interp.integrate1D(u_Λ .^2, Λgrid, [Λgrid[ui], Λgrid[end]])
            _b_deriv = Interp.integrate1D(du_Λ .* b_Λ, Λgrid, [Λgrid[ui], Λgrid[end]])

            # Fs_Λ_new[ui] = -Rex(_Fs, k)/2.0 + _a + _b*u_Λ[ui] + _b_deriv -_c
            Fs_Λ_new[ui] = -Rex(_Fs, k) + _a  

            para_new = UEG.ParaMC(rs=rs, beta=beta, mass2=mass2, Fs=Fs_Λ_new[ui], Fa=-0.0, isDynamic=true) 
            u_Λ_new[ui] = RG.u_from_f(para_new, k, para.kF)[1]
        end
        diff_u = maximum(abs.(u_Λ_new - u_Λ))
        diff_Fs = maximum(abs.(Fs_Λ_new - Fs_Λ))
        println("iter $i, diff_u = $diff_u, diff_Fs = $diff_Fs")
        for li in eachindex(Λgrid)
            # println(Λgrid[li]/para.kF, " : ", Fs_Λ[li], " -> ", Fs_Λ_new[li], " = diff ", Fs_Λ_new[li]-Fs_Λ[li], " with ", vs2[end, li])
            println(Λgrid[li]/para.kF, " : ", Fs_Λ[li], " -> ", Fs_Λ_new[li], " = diff ", Fs_Λ_new[li]-Fs_Λ[li], " with ", a(Fs_Λ_new[li], Λgrid[li], vs2))
        end
        # println("before: ", Fs_Λ)
        # println("after: ", Fs_Λ_new)
        # println("diff: ", Fs_Λ_new-Fs_Λ)
        if diff_Fs/maximum(abs.(Fs_Λ)) < 1e-3 
            break
        end
        u_Λ .= u_Λ *mix .+ u_Λ_new*(1-mix)
        Fs_Λ .= Fs_Λ*mix .+ Fs_Λ_new*(1-mix)
    end
    return u_Λ, Fs_Λ, du_Λ
end

dz = get_z()
ver3 = get_ver3()
print_ver3(ver3; nsample =10)
vuu, vud = get_ver4(dz)
print_ver4(vuu, vud, 1; nsample=10)
print_ver4(vuu, vud, 2; nsample=10)
print_ver4(vuu, vud, 3; nsample=10)
print_ver4(vuu, vud; nsample = 10)

_u, _Fs, _du = solve_RG(vuu, vud, ver3)
println(_Fs[1])
println(_u[1])
