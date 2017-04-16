# computing regularisation of w, add more if you like

function regFunc(w, regType, sn=1e-8)

    if regType == "noreg"
        regV  = 0
        regDV = 0
    elseif regType == "lasso"
        regV  = sqrt(w.^2 + sn)
        regDV = w ./ sqrt(w.^2 + sn)
    elseif regType == "l2"
        regV  = sum(w.^2)
        regDV = 2 .* w
    end

    return regV, regDV

end
