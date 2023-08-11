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

function mu(data)
    return real(data[1, 1])
end

function zCT(para, filename; Fs=fdict[para.rs], Λgrid=SparseΛgrid(para.kF))
    # println("read Fs = $Fs from $filename")
    f = jldopen(filename, "r")
    # z1 = zeros(Measurement{Float64}, length(Fs), length(Λgrid))
    sw = Dict()
    mu = Dict()
    partition = UEG.partition(para.order)
    for p in partition
        sw[p] = MeshArray(Fs, Λgrid; dtype=Measurement{Float64})
        mu[p] = 0.0 # we don't need mu for now
    end
    for (fi, F) in enumerate(Fs)
        _para = get_para(para, F)
        key = UEG.short(_para)
        ngrid, kgrid, sigma = f[key]
        println(length(kgrid), ", ", length(Λgrid))
        @assert kgrid ≈ Λgrid "length(kgrid) = $(length(kgrid)), length(Λgrid) = $(length(Λgrid))"
        for p in partition
            sw[p][fi, :] = zfactor(sigma[p], _para.β)
        end
    end

    dzi, dmu, dz = CounterTerm.sigmaCT(para.order, mu, sw)
    return dzi, dmu, dz
end