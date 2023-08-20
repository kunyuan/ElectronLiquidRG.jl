
fdict = Dict()
fdict[1.0] = collect(LinRange(-0.4, 0.0, 19))
fdict[5.0] = collect(LinRange(-2.0, 0.0, 8))

Λgrid(kF) = CompositeGrid.LogDensedGrid(:uniform, [1.0 * kF, 100 * kF], [kF,], 8, 0.5 * kF, 8)

# SparseΛgrid(kF) = CompositeGrid.LogDensedGrid(:gauss, [1.0 * kF, 32 * kF], [kF,], 4, 0.1 * kF, 4)
SparseΛgrid(kF) = CompositeGrid.LogDensedGrid(:uniform, [1.0 * kF, 16 * kF], [kF,], 4, 0.1 * kF, 4)

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

"""
    function KO(para::ParaMC, kamp=para.kF, kamp2=para.kF; N=100, mix=1.0, verbose=1, ct=false)
    
    Calculate Fs with the self-consistent equation
    ```math
    f_s = -\\left< (v_q+f_s)/(1-(v_q+f_s)Π_0) \\right>
    ```
"""
function KO(para::ParaMC, kamp=para.kF, kamp2=para.kF; a_s=0.0, N=100, mix=0.8, verbose=0, eps=1e-5, spin_spin=false, Fp=-0.2 * para.rs + 4π / para.me * a_s, Fm=0.0)

    u = 0.0

    function _u(fs)
        p_l = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=fs, Fa=0.0, order=1, mass2=para.mass2, isDynamic=true, isFock=false)
        wp, wm, angle = Ver4.exchange_interaction(p_l, kamp, kamp2; ct=false, verbose=verbose)
        return Ver4.Legrendre(0, wp, angle) + fs - 4 * π / para.me * a_s
    end

    function _u_f(fs)
        p_l = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=fs, Fa=0.0, order=1, mass2=para.mass2, isDynamic=true, isFock=false)
        wp, wm, angle = Ver4.exchange_interaction_df(p_l, kamp, kamp2; ct=false, verbose=verbose)
        return (Ver4.Legrendre(0, wp, angle)) / para.NF + 1.0
    end

    function newtown(fs)
        iter = 1
        err = eps * 10
        while err > eps && iter < N
            if verbose > 1
                println("$fs ->", fs - _u(fs) / _u_f(fs), " with ", _u(fs), ", ", _u_f(fs))
            end
            fx = _u(fs)
            fs_new = fs - fx / _u_f(fs)
            # err = abs(fs - fs_new)
            err = abs(fx)
            fs = fs_new
            iter += 1
            if iter >= N
                @warn("Newton-Raphson method doesn't converge. error = $err, got $fs")
            end
        end
        return fs
    end

    if spin_spin == false
        _Fs = newtown(Fp)
        p_l = UEG.ParaMC(rs=para.rs, beta=para.beta, Fs=_Fs, Fa=0.0, order=1, mass2=para.mass2, isDynamic=true, isFock=false)
        wp, wm, angle = Ver4.exchange_interaction(p_l, kamp, kamp2; ct=false, verbose=verbose)
        u = Ver4.Legrendre(0, wp, angle)
        if verbose > 0
            println("Self-consistent approach: ")
            println("Fs = ", _Fs)
            println("Fa = ", 0.0)
            println("u = ", u)
            println("4π a_s/m = ", 4π / para.me * a_s)
            # @assert abs(u + _Fs - 4π / para.me * a_s) < 10 * eps "u is not consistent with 4π a_s/m with error $(u + _Fs - 4π / para.me * a_s)"
        end
        return _Fs, 0.0, u
    else
        @assert spin_spin == false "Spin-Spin KO interaciton doesn't work yet."
    end
end

@inline function linear2D(data, xgrid, ygrid, x, y)

    xarray, yarray = xgrid.grid, ygrid.grid

    xi0, xi1, yi0, yi1 = 0, 0, 0, 0
    if (x <= xarray[firstindex(xgrid)])
        xi0 = 1
        xi1 = 2
    elseif (x >= xarray[lastindex(xgrid)])
        xi0 = lastindex(xgrid) - 1
        xi1 = xi0 + 1
    else
        xi0 = floor(xgrid, x)
        xi1 = xi0 + 1
    end

    if (y <= yarray[firstindex(ygrid)])
        yi0 = 1
        yi1 = 2
    elseif (y >= yarray[lastindex(ygrid)])
        yi0 = lastindex(ygrid) - 1
        yi1 = yi0 + 1
    else
        yi0 = floor(ygrid, y)
        yi1 = yi0 + 1
    end

    dx0, dx1 = x - xarray[xi0], xarray[xi1] - x
    dy0, dy1 = y - yarray[yi0], yarray[yi1] - y

    g0 = data[xi0, yi0] * dx1 + data[xi1, yi0] * dx0
    g1 = data[xi0, yi1] * dx1 + data[xi1, yi1] * dx0

    gx = (g0 * dy1 + g1 * dy0) / (dx0 + dx1) / (dy0 + dy1)
    return gx
end

function linear_interp(Gf, Fmesh, Kmesh, Fs, k)
    # println("GF: ", Gf)
    if k > Kmesh[end]
        return zero(eltype(Gf.data))
    end
    # @assert Fs <maximum(Fmesh) "Fs $Fs is larger than maximum of Fmesh"
    # @assert Fs > minimum(Fmesh) "Fs $Fs is smaller than minimum of Fmesh"

    return linear2D(Gf.data, Fmesh, Kmesh, Fs, k)
end