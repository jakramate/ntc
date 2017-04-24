# ==========================================================================
# Name  :  Robust Logistic Regression
# Author:  Jakramate Bootkrajang
# Last update: 16 April 2013
# Input :
#       w = given initial model's parameters
#       g = initial label flipping probability matrix
#       x = distesign matrix where a row represents a sample
#       y = Target value
#       options.maxIter = maximum optimising iteration
#       options.estGam  = estimating the gamma using multiplicative update
#       options.regFunc = type of regularisation: 'noreg', 'L1' or 'L2'
#       options.boost   = adapting to boosting framework
#       options.sn      = Small number for approximating L1 objective.
#       options.verbose = displaying negative log-likelihood
# Output:
#       w   = fitted model
#       g   = estimated gamma matrix
#       llh = negative log-likelihood
# Note  :
#       1. The function uses {0,1} class representation.
#       w = (eye(size(x,2))-x'*inv(x*x'+eye(size(x,1)))*x)*x'*y/2;  w(1) = 1;
#  |-----|-----------|
#  |     |  0   y^  1|
#  ------|-----------|
#  |y  0 | g00   g01 |
#  |   1 | g10   g11 |
#  |-----|-----------|
# ==========================================================================
# ==========================================================================

function rlr(w, g, x, y; sn=1e-8, maxIter=50, dist=1, estG=true, regType="noreg")

  # getting to know your data
  ndata, dim = size(x)

  # ensure that y is {0,1}
  y = castLabel(y,0)

  # ensure that the problem is really a binary problem
  if (size(unique(y),1) != 2)
    error("Trying to solve multiclass problem")
  end

  # it's good to add bias term to the input vector
  if (sum(x[:,1]) != ndata)
    display("Bias term might have not been added")
  end

  if size(dist,1) != ndata
    dist = ones(ndata,1)
  end

  # storage for likelihood values
  llh = zeros(1,maxIter)

  for l=1:maxIter

      # calculating regularisation parameter lambda
      if l == 1
          lambda = regParam(w, "noreg", sn)     # no regularisation on the first run
      else
          lambda = regParam(w, regType, sn)
      end

      t = x * w

      if estG
          # using multiplicative update
          s0  = (g[1,1] * (1 ./ ((1 ./ exp(-t))+1))) + (g[2,1] ./ (1+exp(-t)))
          s1  = (g[1,2] * (1 ./ ((1 ./ exp(-t))+1))) + (g[2,2] ./ (1+exp(-t)))

          # avoiding numerical problem
          s0[s0.==0] = sn
          s1[s1.==0] = sn

          g00 = g[1,1] * sum((dist.*(1 - y) ./ (s0)) .* (1 ./ ((1 ./ exp(-t))+1)),1)
          g01 = g[1,2] * sum((dist.*y ./ s1) .* (1 ./ ((1 ./ exp(-t))+1)),1)


          # (g00 + g01) is the Lagrangian
          g11 = vec(g00 / (g00 + g01))

          g[1,1] = g11[1,1]
          g[1,2] = 1 - g[1,1]

          g10 = g[2,1] * sum((dist.*(1-y) ./ s0) ./(1+exp(-t)),1)
          g11 = g[2,2] * sum((dist.*(y) ./ (s1)) ./(1+exp(-t)),1)

          # (g10 + g11) is the Lagrangian
          g21 = vec(g10 / (g10 + g11))
          g[2,1] = g21[1,1]
          g[2,2] = 1 - g[2,1]

      end

      # optimising the weight vector
      w, fx, iter = minimize(w, gradw, 10, g, x, y, lambda, dist, regType)

      llh[l] = fx[end]
  end
  
  return w, g, llh

end


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
