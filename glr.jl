# ==========================================================================
# Name  :  Robust Logistic Regression with non-random label noise model
# Author:  Jakramate Bootkrajang
# Last update: 7 September 2015
# Input :
#       w = given initial model's parameters
#       g = initial label flipping probability matrix
#       x = Design matrix where a row represents a sample
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
#       w = (eye(size(x,2))-x'*inv(x*x'+eye(size(x,1)))*x)*x'*y/2  w(1) = 1
#  |-----|-----------|
#  |     |  0   y^  1|
#  ------|-----------|
#  |y  0 | g00   g01 |
#  |   1 | g10   g00 |
#  |-----|-----------|
# ==========================================================================
# ==========================================================================:w

function glr(w, x, y; sn=1e-8, maxIter=50, estG=true, regType="noreg")
    
    ndata, dim = size(x)
    
    # The function uses {0,1} class representation.
    y = castLabel(y,0)

    ## Some input preprocessing
    # check if the problem is binary problem
    if (size(unique(y),1) != 2) 
        error("Trying to solve multiclass problem")
    end

    # checking if bias term is added i.e. first
    # column must be unique and equals to 1
    if (sum(x[:,1]) != ndata)
        display("Bias terms might have not been added")
    end

    # a variable for storing log-likelihood values
    llh  = zeros(1,maxIter)
    # parameters to the gamma function
    t0 = [500]
    t1 = [500]
    t  = x * w     
    z  = t./norm(w)
    lambda = regParam(w, "noreg", sn)
    
    ## ========================= BEGIN ESTIMATING PARAMETERS ===========================
    for l=1:maxIter 
        
        bReg0 = -1 ./(1-t0)
        bReg1 = -1 ./(1-t1)
    
        # no regularisation and use uniform noise on the first iter
        if l==1
            g01    = ones(size(y)) * 0.2
            g10    = ones(size(y)) * 0.2
        else        
            g01    = gammapdf(z,1,t0)
            g10    = gammapdf(z,1,t1)
        end
    
         
        # optimising the weight vector         
        w, fw, i = minimize(w, gradw_glr, 5, g01, g10, x, y, bReg0, bReg1, lambda, regType)

        # update z based on new weight vector
        t = x * w     
   
        z = t./norm(w)
    
        # recomputing the regularisation term based on new w
        lambda = regParam(w, regType, sn)

        # calculating regularisation for w
        regV, regDV = regFunc(w, regType, sn)
        regW        = sum(lambda .* regV)
         
        # estimating noise parameters
        if estG
            t0, fx, iter = minimize(t0, gradt0, 3, z, t, y, t1, regW, bReg0, bReg1)
            t1, fx, iter = minimize(t1, gradt1, 3, z, t, y, t0, regW, bReg0, bReg1)
        end

        llh[l] = fx[end]
    end
    ## ======================== END ESTIMATING PARAMETER =======================

    return w, t0, t1, llh

end



# function definition for 'minFunc' optimiser
# fv  = function value
# dfv = gradient of the function w.r.t W
# for non-uniform noise rate model

function gradw_glr(w, g01, g10, x, y, bReg0, bReg1, lambda, regType)

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


# computing gradient of the nlr's objective w.r.t
# the parameter of the gamma's PDF
function gradt1(t1, z, t, y, t0, regW, bReg0, bReg1)

    g01 = gammapdf(z,1,t0)
    g10 = gammapdf(z,1,t1)
    g00 = 1 - g01
    g11 = 1 - g10

    p1 = 1 ./ (1+exp(-t))
    p0 = 1-p1

    # computing function value
    s0 = (g00 .* p0) + (g10 .* p1)
    s1 = (g01 .* p0) + (g11 .* p1)

    s0[s0.==0] = eps() 
    s1[s1.==0] = eps()

    fv = -sum((y .* log(s1)) + ((1 - y).* log(s0)),1) + regW + bReg0 .* log(t0-1) + bReg1.*log(t1-1)

    # computing corresponding derivative
    tmp1  = ((1 - y) ./ s0) - (y ./ s1)
    tmp2  = ((g10 .* z ./ t1.^2) - (g10 ./t1))

    gAux1 = (tmp1 .* tmp2) .* p1
    dfv   = -sum(gAux1,1)' + bReg1 ./ log(t1-1)

    return fv[1], dfv

end


# computing gradient of nlr objective function w.r.t. gamma PDF's parameter
#
#
function gradt0(t0, z, t, y, t1, regW, bReg0, bReg1)

    g01 = gammapdf(z,1,t0)
    g10 = gammapdf(z,1,t1)
    g00 = 1 - g01
    g11 = 1 - g10

    p1  = 1 ./ (1+exp(-t))
    p0  = 1 - p1

    # computing function value
    s0  = (g00 .* p0) + (g10 .* p1)
    s1  = (g01 .* p0) + (g11 .* p1)

    s0[s0.==0] = eps() 
    s1[s1.==0] = eps()

    fv = -sum((y .* log(s1)) + ((1 - y).* log(s0)),1) + regW + bReg0.*log(t0-1) + bReg1.*log(t1-1)

    # computing corresponding derivative
    tmp1  = ((y ./ s1) - ((1 - y) ./ s0))
    tmp2  = ((g01 .* z ./ t0.^2) + (g01 ./t0))

    gAux1  = (tmp1 .* tmp2) .* p0
    dfv    = sum(gAux1,1)' + bReg0 ./ log(t0-1)


    return fv[1], dfv

end


