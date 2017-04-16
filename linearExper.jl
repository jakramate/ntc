using MAT    # for reading Matlab's file

# adding libraries here
include("robust_lr/rlr.jl")
include("robust_nlr/nlr.jl")
include("robust_hlr/hlr.jl")
include("utils/addbias.jl")
include("utils/standardise.jl")
include("utils/castLabel.jl")
include("utils/injectLabelNoise.jl")
include("optimiser/minimize.jl")
include("regulariser/regFunc.jl")
include("regulariser/regParam.jl")

# some constants
repetitions = 10

# reading data, currently in MATLAB format
dataset = ["websearch","boston","australia"]

for d in dataset

    vars = matread("./datasets/$d.mat")
    x = vars["x"]
    y = vars["y"]
    dpnt, dim = size(x)
    tpnt = convert(Integer,floor(0.8 * dpnt))

    err_rlr = zeros(repetitions,1)
    err_glr = zeros(repetitions,1)
    err_hlr = zeros(repetitions,1)

    for i=1:repetitions
        
        # random train/test split
        perm =randperm(dpnt)
        xt = x[perm[1:tpnt],:]
        yt = y[perm[1:tpnt]]
        xs = x[perm[tpnt+1:end],:]
        ys = y[perm[tpnt+1:end]]

        xt, xs = standardise(xt, xs)

        xt = addbias(xt)
        xs = addbias(xs)
        winit = rand(dim+1,1)

        yz, fd = injectLabelNoise(yt, [0.0 0.0])

        # robust logistic regression
        w, g, llh = rlr(winit, [0.8 0.2;0.2 0.8], xt, yz, regType="noreg")
        err_rlr[i] = sum(sign(xs * w) .!= castLabel(ys,-1))/length(ys)
        #err_rlr[i] = 0

        # generalised robust logistic regression
        #wg, t0, t1, llhg = nlr(winit, xt, yz, regType="l2")
        #err_glr[i] = sum(sign(xs * wg) .!= castLabel(ys,-1))/length(ys)
        err_glr[i] = 0

        # hybrid robust logistic regression
        #wh, mix, llhh = hlr(winit, xt, yz, regType="l2", mixCom=3, mixCov="diag")
        #err_hlr[i] = sum(sign(xs * wh) .!= castLabel(ys,-1))/length(ys)
        err_hlr[i] = 0

        @printf("[Round %3d] rlr = %.2f, glr = %.2f, hlr = %.2f\n", 
                i, err_rlr[i], err_glr[i], err_hlr[i])
  
    end

    @printf("[%c%s] rlr = %.2f, glr = %.2f, hlr=%.2f \n", 
            uppercase(d[1]),d[2:end], mean(err_rlr), mean(err_glr), mean(err_hlr))


end
