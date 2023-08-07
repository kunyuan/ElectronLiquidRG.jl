function vertex4(para; neval=1e6, Λgrid=Λgrid(para.kF), n=[-1, 0, 0, -1], l=[0,], filename=nothing)
    ver4, result = Ver4.MC_PH(para; kamp=Λgrid, n=n, l=l, neval=neval, filename=filename, filter=[NoHartree, NoBubble, Proper], channel=[PHr, PHEr, PPr])
    return ver4, result
end