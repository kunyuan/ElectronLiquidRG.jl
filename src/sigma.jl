function sigma(para; neval=1e6, Λgrid=Λgrid(para.kF), filename=nothing)
    sigma, result = Sigma.MC(para; kgrid=Λgrid, ngrid=[-1, 0, 1], neval=neval, filename=filename)
    return sigma, result
end

function zfactor(data, β, ngrid=[-1, 0]) #assume the data are calculated with [-1, 0, 1]
    if ngrid == [0, 1]
        return @. (imag(data[3, :]) - imag(data[2, :])) / (2π / β)
    elseif ngrid == [-1, 0]
        return @. (imag(data[2, :]) - imag(data[1, :])) / (2π / β)
    else
        error("ngrid = $ngrid not implemented")
    end
end

function zCT(para, filename; Fs=fdict[para.kF], Λgrid=Λgrid(para.kF))
    f = jldopen(filename, "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))
    z1 = MeshArray(Fs, Λgrid; dtype=Measurement{Float64})
    for (fi, F) in enumerate(Fs)
        _para = get_para(para, F)
        key = "$(UEG.short(_para))"
        ngrid, kgrid, sigma = f[key]
        @assert kgrid ≈ Λgrid
        z1[fi, :] = zfactor(sigma[(1, 0, 0)], _para.β)
    end
    return z1
end