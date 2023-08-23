using ElectronLiquidRG
using ElectronLiquid

neval = 4e9

dim = 3
rs = [5.0,]
mass2 = [0.001,]
beta = [25.0]
order = [1,]
isDynamic = true
isFock = false


for (_rs, _mass2, _beta, _order) in Iterators.product(rs, mass2, beta, order)
    # for _F in ElectronLiquidRG.fdict[_rs]
    # for _F in [ -0.3, -0.2, -0.1, 0.0]
    # for _F in [-1.2, -1.0, -0.8, -0.6, -0.4]
    for _F in [-1.0, -0.8, -0.6, -0.4]
    # for _F in [-2.0, -1.8, -1.6, -1.4, -1.2]
    # for _F in [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]
        # for _F in [0.0, ]
        para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)
        kF = para.kF

        Λgrid = ElectronLiquidRG.SparseΛgrid(kF)
        # Λgrid = [para.kF,]

        println("Sigma on $(UEG.short(para))")
        ElectronLiquidRG.sigma(para, Λgrid=Λgrid, neval=neval, filename="data/sigma.jld2")

        para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order+1, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)

        println("Ver4 on $(UEG.short(para))")
        ver4, res = ElectronLiquidRG.vertex4(para, Λgrid=Λgrid, neval=neval, filename="data/ver4.jld2")
        if isnothing(ver4) == false
            println(ver4[(2, 0, 0)][2, 1, 1])
        end

        println("PP ver3 on $(UEG.short(para))")
        ElectronLiquidRG.vertex3(para, kamp=Λgrid, neval=neval, integrand=ElectronLiquidRG._PP, filename="data/ver3_PP.jld2")

        println("PHE ver3 on $(UEG.short(para))")
        ElectronLiquidRG.vertex3(para, kamp=Λgrid, neval=neval, integrand=ElectronLiquidRG._Lver3, filename="data/ver3_PHE.jld2")

        println("PH ver3 on $(UEG.short(para))")
        ElectronLiquidRG.vertex3(para, kamp=Λgrid, neval=neval, integrand=ElectronLiquidRG._Lbubble, filename="data/ver3_PH.jld2")
    end
end