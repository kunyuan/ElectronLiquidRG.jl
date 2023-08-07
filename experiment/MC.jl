using ElectronLiquidRG
using ElectronLiquid

neval = 1e6

dim = 3
rs = [1.0,]
mass2 = [0.01,]
# Fs = [-0.0,]
beta = [25.0]
order = [1,]
neval = 1e6
# neval = 1e6
isDynamic = true
isFock = false


for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
    for _F in ElectronLiquidRG.fdict[_rs]
        para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)
        kF = para.kF

        Λgrid = ElectronLiquidRG.Λgrid(kF)

        println("Sigma on $(UEG.short(para))")
        ElectronLiquidRG.sigma(para, Λgrid=Λgrid, neval=neval, filename="data/sigma.jld2")

        println("Ver4 on $(UEG.short(para))")
        ElectronLiquidRG.vertex4(para, Λgrid=Λgrid, neval=neval, filename="data/ver4.jld2")
    end
end