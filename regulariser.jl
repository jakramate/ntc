function regParam(w, regType, sn=1e-8)

    lambda = zeros(size(w,1),1)

    if regType == "lasso"
        lambda[:] = 1 ./ sqrt(w.^2 + sn)
    elseif regType == "l2"
        lambda[:] = (length(w)/2 + 1) / sum(w[2:end].^2/2 + 2)
    elseif regType == "noreg"
        lambda[:] = 0
    end


    lambda[1] = 0

    return lambda

end
