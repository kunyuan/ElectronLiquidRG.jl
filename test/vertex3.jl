@testset "vertex3" begin
    para = ParaMC(rs=1.0, dim=3, beta=25.0, mass2=0.001, Fs=-0.0, Fa=-0.0, isDynamic=true, order=1);
    k = 20.0*para.kF
    data, res = ElectronLiquidRG.vertex3(para; kamp = [k, ], neval = 1e6)
    expect = -para.me*para.e0^2/(2*k)*π*para.NF #minus sign is from the direct Coulomb interaction
    println(data, " vs ", expect)
    @test abs(real(data[1]).val - expect) <10*real(data[1]).err
end