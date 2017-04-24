function addbias(x)
  return hcat(ones(size(x,1),1),x)
end


# Name  : A function to inject label noise
# Author: Jakramate Bootkrajang
# Input : y           = true label {1,...n} representation
#         targetClass = {0,1,2....} s.t. 0 is symmetric NoiseRateing
#         NoiseRate   = (0,1)
# Output: yz          = noisy label  {1,...n} representation
#         fp          = flip indicator vector


function injectLabelNoise(y, flipRate)

    fd  =  -ones(size(y))
    yz  =  castLabel(y,-1)
    y   =  castLabel(y,2)

    # sampling some numbers
    for i=1:2
        prob    = rand(size(y))   
        idx     = find((y.==i) & (prob .<= flipRate[i]))
        yz[idx] = -yz[idx]
        fd[idx] = -fd[idx]
    end

    yz = castLabel(yz,2)

    return yz, fd

end


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


# Description
#        Function to cast between different type of
#        label representation i.e. from {-1,1} to {0,1} to {1,2}
# Author
#        Jakramate Bootkrajang
# Input
#        y = label vector
#        t = target representation choose from {-1,0,2}
# Output
#        y = casted label vector
# ==========================================================================

function castLabel(y, t)

  if -1 in y
    if t == -1
      y_new = y;
    elseif t == 0
      y_new = (y + 1) ./ 2
    elseif t == 2
      y_new = (y + 3) ./ 2
    end
  elseif 0 in y
    if (t == -1)
      y_new = y .* 2 - 1
    elseif (t == 0)
      y_new = y
    elseif (t == 2)
      y_new = y + 1
    end
  elseif 2 in y
    if (t == -1)
      y_new = y .* 2 - 3
    elseif (t == 0)
      y_new = y - 1
    elseif (t == 2)
      y_new = y
    end
  end

  return y_new
end





# Probability density function of the Gamma distribution

function gammapdf(x,k,theta)
    
    x = abs(x)
    y = ((x.^(k-1)).*exp(-x/theta))/(theta^k * factorial(k-1))

    return y
end

