using ElectronLiquidRG
using ElectronLiquid

neval = 1e7

dim = 3
rs = [5.0,]
mass2 = [0.001,]
beta = [25.0]
order = [1,]
isDynamic = true
isFock = false


for (_rs, _mass2, _beta, _order) in Iterators.product(rs, mass2, beta, order)
    for _F in ElectronLiquidRG.fdict[_rs]
    # for _F in [0.0, ]
        para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)
        kF = para.kF

        Λgrid = ElectronLiquidRG.Λgrid(kF)
        # Λgrid = [para.kF,]

        println("Sigma on $(UEG.short(para))")
        ElectronLiquidRG.sigma(para, Λgrid=Λgrid, neval=neval, filename="data/sigma.jld2")

        para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order+1, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)

        println("Ver4 on $(UEG.short(para))")
        ElectronLiquidRG.vertex4(para, Λgrid=Λgrid, neval=neval, filename="data/ver4.jld2")
    end
end