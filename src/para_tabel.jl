
fdict = Dict()
fdict[1.0] = collect(LinRange(-1.0, 0.0, 16))
fdict[5.0] = collect(LinRange(-2.0, 0.0, 8))

Λgrid(kF) = CompositeGrid.LogDensedGrid(:gauss, [1.0 * kF, 100 * kF], [kF,], 8, 0.01 * kF, 8)

get_para(para, Fs) = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=Fs, Fa=-0.0, order=para.order,
    mass2=para.mass2, isDynamic=true, isFock=false)

function u_from_f(para::ParaMC, kamp=para.kF, kamp2=kamp; verbose=0, N=32)
    c = 2 * para.kF / para.qTF
    Δ = 1 - 3 * c^2 * (1 + para.Fs)
    if (para.Fs < -0.0 && Δ > 0.0) || (para.Fs > 0.0 && Δ < 0.0)
        # here we approximate the Lindhard function with 1-x^2/3 where x=|q|/2kF
        # the most dangerous q is given by the condition that the derivative of the denorminator is zero
        # K[x]=(1+Fs*(1-x^2/3))*c^2*x^2 + (1-x^2/3)
        # dK[x]/dx = 2/3*x*(-1+c^2(3+Fs(3-2x^2)))
        # the only possible solution is x = sqrt((1-3c^2*(1+Fs))/(-2c^2*Fs)) 
        if 3 * c^2 * (1 + para.Fs) < 1.0
            x = sqrt(abs(Δ / 2 / para.Fs / c^2))
            #println("solution: ", x)
            if x * para.kF / kamp > 1.0
                x = 1.0
            end
            theta = asin(x) * 2.0
            #println(x, "--> ", theta)
            θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, theta, π], 16, 0.001, N)
        else
            θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 0.001, N)
        end
    else
        θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 0.001, N)
    end
    wp, wm, angle = Ver4.exchange_interaction(para, kamp, kamp2; ct=false, verbose=verbose, θgrid=θgrid)
    # assert all wp elements are positive
    # @assert all(wp .> 0.0) "wp grid is not positive"
    return Ver4.Legrendre(0, wp, angle) + para.Fs, θgrid, wp
end

function linear_interp(Gf, Fs, k)
    return UEG.linear2D(Gf.data, Gf.mesh[1], Gf.mesh[2], Fs, k)
end