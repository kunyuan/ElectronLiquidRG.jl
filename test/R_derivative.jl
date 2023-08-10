@testset "RG_treelevel_interaction" begin
    para = UEG.ParaMC(rs=4.0, beta=100.0, order=1, mass2=1e-5, isDynamic=true, isFock=false, Fs=-0.0, Fa=-0.0)

    fs = collect(LinRange(0.0, -0.5, 1000))
    dfs = fs[2]-fs[1]
    # fs = CompositeGrid.Uniform{Float64}([0.0, 0.5], 100)

    paras = [UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=fs[li], Fa=0.0, order=para.order,
        mass2=para.mass2, isDynamic=true, isFock=false) for li in eachindex(fs)]

    # d R_ex/df on FS
    dR = [ElectronLiquidRG.∂R_∂F_exchange(p, para.kF, para.kF; ct=false) for p in paras]

    # R(f=0) - R(f=f_0)
    diff = sum(dR)*dfs
    # diff = Interp.integrate1D(dR, fs)

    wp, wm, angle = Ver4.exchange_interaction(paras[1], para.kF, para.kF; ct=false)
    W0 = Ver4.Legrendre(0, wp, angle)

    wp, wm, angle = Ver4.exchange_interaction(paras[end], para.kF, para.kF; ct=false)
    R_f = Ver4.Legrendre(0, wp, angle)

    println(" W0: ", W0, " R_f: ", R_f, " diff: ", diff)
    @test abs(diff - (R_f-W0)) < 1e-3

    ########### test kgrid ##########################################
    Λgrid = CompositeGrid.LogDensedGrid(:gauss, [1.0 * para.kF, 100 * para.kF], [para.kF,], 8, 0.01 * para.kF, 8)

    para = UEG.ParaMC(rs=4.0, beta=100.0, order=1, mass2=1e-5, isDynamic=true, isFock=false, Fs=-0.5, Fa=-0.0)


    dR = [ElectronLiquidRG.∂R_∂Λ_exchange(para, l, para.kF; ct=false) for l in Λgrid]
    diff = Interp.integrate1D(dR, Λgrid)

    wp, wm, angle = Ver4.exchange_interaction(para, Λgrid[1], para.kF; ct=false)
    W0 = Ver4.Legrendre(0, wp, angle)

    wp, wm, angle = Ver4.exchange_interaction(para, Λgrid[end], para.kF; ct=false)
    Wf = Ver4.Legrendre(0, wp, angle)

    println(" W: ", W0, " -> ", Wf, " diff: ", diff)
    @test abs(diff - (Wf-W0)) < 1e-3
end