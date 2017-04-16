# Name  :  Hybrid Robust Logistic Regression
# Author:  Jakramate Bootkrajang
# Last update: 16 April 2013
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
include("gmm.jl")
include("lmm.jl")
include("gradw_hr.jl")
include("calActiv.jl")
include("aposterior.jl")
include("../density/gmmpdf.jl")

function hlr(w, x, y; mixCom=3, mixCov="diag", sn=1e-8, maxIter=50, estG=true, regType="noreg")

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

    # for storing log-likelihood values
    llh = zeros(1, maxIter)

    # initialising mixture model here
    # gmm's dimension must match z's dimension
    mix = [1=>0,2=>0]
    for i=1:2
        mix[i] = gmm(2, mixCom, mixCov)    
    end


    ## ========================= BEGIN ESTIMATING PARAMETERS ===========================
    for l=1:maxIter
    
        if l==1
            lambda = regParam(w, "noreg", sn)
            g01    = ones(size(y)) * 0.2
            g10    = ones(size(y)) * 0.2
        else
            g01 = gmmpdf(z, mix[1])
            g10 = gmmpdf(z, mix[2])
        
            g01 = g01/norm(g01)
            g10 = g10/norm(g10)
        end
   
    
        # optimising the weight vector
        w, fw, i = minimize(w, gradw_hr, 5, g01, g10, x, y, lambda, regType)
    
        # calculating likelihood function
        llh[l] = fw[end]
    
        # update z based on new weight vector
        t = x * w

        # stack your indicators here
        # [dis_from_boundary, cosine_sim, ....]
        z1 = t/norm(w)
        z2 = t/(norm(w)*norm(x))
        z  =  [z1 z2] 

        # set z to 0 for those with correct prediction
        z  =  z .* (sign(t) .!= castLabel(y,-1))        
    
        # recomputing the regularisation term based on new w
        lambda = regParam(w, regType, sn)

        # calculating regularisation for w
        regV, regDV  = regFunc(w, regType, sn)
        regW         = sum(lambda .* regV)
    
        # learning mixture model here 
        if estG                        
            # checking for misclassified points which are indicated by the
            # first feature of 'z'
 
            z_pos = z[z[:,1] .> 0,:]
            z_neg = z[z[:,1] .< 0,:]
        
            if size(z_pos,1)>0
                mix[2] = lmm(z_pos, mix[2])
            end
        
            if size(z_neg,1)>0
                mix[1] = lmm(z_neg, mix[1])
            end      

        end
    end

    ## ======================== END ESTIMATING PARAMETER =======================
    

    return w, mix, llh

end

