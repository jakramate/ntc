# function definition for 'minFunc' optimiser
# fv  = function value
# dfv = gradient of the function w.r.t W
# for non-uniform noise rate model

function gradw_nr(w, g01, g10, x, y, bReg0, bReg1, lambda, regType)

    regV, regDV = regFunc(w, regType, 1e-8)

    t  =  x * w

    s0 = ((1-g01) .* (1 ./ ((1 ./ exp(-t))+1))) + (g10 ./ (1+exp(-t)))
    s1 = (g01 .* (1 ./ ((1 ./ exp(-t))+1))) + ((1-g10) ./ (1+exp(-t)))

    s0[s0.==0] = eps() 
    s1[s1.==0] = eps()

    # function value
    fv  = -sum((y .* log(s1)) + ((1 - y) .* log(s0)),1) + sum(lambda .* regV) + bReg0 + bReg1
    
    # compute corresponding derivative
    tmp1 = ((1-g10) - g01) .* (y ./ s1)
    tmp2 = (g10 - (1-g01)) .* ((1 - y) ./ s0)


    gAux1 = (tmp1 + tmp2) ./ (1+exp(-t)) .* (1 ./ ((1 ./ exp(-t))+1))

    # completed Eq.34 with dot product with x_n
    gAux2  = repmat(gAux1, 1, size(x,2)) .* x
    dfv    = -sum(gAux2,1)' + (lambda .* regDV)

    return fv[1], dfv

end
