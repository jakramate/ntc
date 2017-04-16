function gradw(w, gamma, x, y, lambda, dist, regType)

    regV, regDV = regFunc(w, regType, 1e-8)

    t     = x * w

    s0    = (gamma[1,1] * (1 ./ ((1 ./ exp(-t))+1))) + (gamma[2,1] ./ (1+exp(-t)))
    s1    = (gamma[1,2] * (1 ./ ((1 ./ exp(-t))+1))) + (gamma[2,2] ./ (1+exp(-t)))


    s0[s0.==0] = eps() 
    s1[s1.==0] = eps()

    # function value
    fv    = -sum((dist .* y .* log(s1)) + (dist .* (1 - y).* log(s0)),1) + sum(lambda .* regV)

    # compute corresponding derivative
    tmp1  = (gamma[2,2] - gamma[1,2]) * (dist .* y ./ s1)
    tmp2  = (gamma[2,1] - gamma[1,1]) * (dist .* (1 - y) ./ s0)

    gAux1 = (tmp1 + tmp2) ./ (1+exp(-t)) .* (1 ./ ((1 ./ exp(-t))+1))

    # completed Eq.34 with dot product with x_n
    gAux2  = repmat(gAux1, 1, size(x,2)) .* x
    dfv    = -sum(gAux2,1)' + (lambda .* regDV)


    return fv[1], dfv

end
