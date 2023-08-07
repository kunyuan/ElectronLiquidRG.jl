using ElectronLiquid
import ElectronLiquidRG as RG
using JLD2

para = ParaMC(rs=1.0, beta=25.0, mass2=0.01, Fs=-0.0, Fa=-0.0, isDynamic=true, order=1)
z1 = RG.zCT(para, "data/sigma.jld2"; Fs=RG.fdict[para.rs])

println(z1[end, :])
println(z1.mesh[1][end])
println(z1.mesh[2][1])