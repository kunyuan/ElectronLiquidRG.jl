using ElectronLiquidRG
using ElectronLiquid

neval = 1e6

dim = 3
rs = [1.0,]
mass2 = [1.0e-5,]
Fs = [-0.0,]
beta = [25.0]
order = [1,]
neval = 1e6
# neval = 1e6
isDynamic = true
isFock = false


for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
    # for _F in ElectronLiquidRG.fdict[_rs]
    for _F in [0.0,]
        para = UEG.ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock)
        kF = para.kF

        println("working on $(UEG.short(para))")
        ElectronLiquidRG.MC_PH(para, neval=neval, filename="data/ver4.jld2")
    end
end