@testset "Vertex4" begin
    para = ParaMC(rs=1.0, dim=3, beta=25.0, mass2=0.001, Fs=-0.0, Fa=-0.0, isDynamic=true, order=1)
    k = 20.0 * para.kF

    c_pp = ElectronLiquidRG.c_coeff_pp(para, k, para.kF)
    expect = para.me / 8 / π * k / para.NF
    println(c_pp, " vs ", expect)
    @test abs(real(c_pp) - expect) / expect < 1e-2

end