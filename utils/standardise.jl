# make sure that an observation is a row vector


function standardise(x, xt)

    dpnt = size(x,1)
    tpnt = size(xt,1)

    xa = x

    offset = mean(xa,1)
    var    = std(xa,1)

    var[var.==0] = var[var.==0] + 1
    scale = 1./var

    x = x -  (ones(size(x)) .* offset)
    x = x ./ (ones(size(x)) .* scale)
    xt = xt -  (ones(size(xt)) .* offset)
    xt = xt ./ (ones(size(xt)) .* scale)
    

    return x, xt
end
