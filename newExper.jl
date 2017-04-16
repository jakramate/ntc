using MAT    # for reading Matlab's file

# adding libraries here
include("robust_lr/rlr.jl")
include("robust_nlr/nlr.jl")
include("robust_hlr/hlr.jl")
include("utils/addbias.jl")
include("utils/standardise.jl")
include("utils/castLabel.jl")
include("optimiser/minimize.jl")
include("regulariser/regFunc.jl")
include("regulariser/regParam.jl")

# some constants
repetitions = 100

# reading data, currently in MATLAB format
dataset = "adult"
vars = matread("./datasets/$dataset.mat")
x = vars["x"]
y = vars["y"]
dpnt, dim = size(x)
tpnt = round(Int, 0.8 * dpnt)

err_rlr = zeros(repetitions,1)
err_glr = zeros(repetitions,1)
err_hlr = zeros(repetitions,1)

for i=1:repetitions

  # random train/test split
  perm = randperm(dpnt)
  xt   = x[perm[1:tpnt],:]
  yt   = y[perm[1:tpnt]]
  xs   = x[perm[tpnt+1:end],:]
  ys   = y[perm[tpnt+1:end]]

  xt, xs = standardise(xt, xs)

  # robust logistic regression
  w, g, llh = rlr(zeros(dim+1,1), [0.8 0.2;0.2 0.8], addbias(xt), yt, regType="lasso")
  err_rlr[i] = sum(sign(addbias(xs) * w) .!= castLabel(ys,-1))/length(ys)
  #err_rlr[i] = 0

  # generalised robust logistic regression
  wg, t0, t1, llhg = nlr(zeros(dim+1,1), addbias(xt), yt, regType="lasso")
  err_glr[i] = sum(sign(addbias(xs) * wg) .!= castLabel(ys,-1))/length(ys)
  #err_glr[i] = 0

  # hybrid robust logistic regression
  wh, mix, llhh = hlr(zeros(dim+1,1), addbias(xt), yt, regType="lasso", mixCom=1, mixCov="cdiag")
  err_hlr[i] = sum(sign(addbias(xs) * wh) .!= castLabel(ys,-1))/length(ys)

  @printf("[Round %d] rlr = %.2f , glr = %.2f hlr = %.2f\n", i, err_rlr[i], err_glr[i], err_hlr[i])
  @printf("[llh] rlr = %.2f , glr = %.2f hlr = %.2f\n", llh[end], llhg[end], llhh[end])
  
end

@printf("[Mean] rlr = %.2f , glr = %.2f hlr=%.2f \n", mean(err_rlr), mean(err_glr), mean(err_hlr))
