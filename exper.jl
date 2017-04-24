using MAT    # for reading Matlab's file

# adding libraries here
include("rlr.jl")
include("glr.jl")
include("utils.jl")
include("minimize.jl")
include("regulariser.jl")

# some constants
repetitions = 10

# reading data, currently in MATLAB format
dataset = "fertility"
vars    = matread("./datasets/$dataset.mat")
x       = vars["x"]
y       = vars["y"]
dpnt, dim = size(x)
tpnt    = round(Int, 0.8 * dpnt)

err_rlr = zeros(repetitions,1)
err_glr = zeros(repetitions,1)

for i=1:repetitions

  # random train/test split
  perm = randperm(dpnt)
  xt   = x[perm[1:tpnt],:]
  yt   = y[perm[1:tpnt]]
  xs   = x[perm[tpnt+1:end],:]
  ys   = y[perm[tpnt+1:end]]

  xt, xs = standardise(xt, xs)


  yz, fd = injectLabelNoise(yt, [0.8 0.2;0.2 0.8])

  # robust logistic regression
  w, g, llh = rlr(zeros(dim+1,1), [0.8 0.2;0.2 0.8], addbias(xt), yt, regType="lasso")
  err_rlr[i] = sum(sign(addbias(xs) * w) .!= castLabel(ys,-1))/length(ys)

  # generalised robust logistic regression
  wg, t0, t1, llhg = glr(zeros(dim+1,1), addbias(xt), yt, regType="lasso")
  err_glr[i] = sum(sign(addbias(xs) * wg) .!= castLabel(ys,-1))/length(ys)


  @printf("[Round %d] rlr = %.2f , glr = %.2f\n", i, err_rlr[i], err_glr[i])
  @printf("[log-likelihood] rlr = %.2f , glr = %.2f\n", llh[end], llhg[end])
  
end

@printf("[Mean] rlr = %.2f , glr = %.2f \n", mean(err_rlr), mean(err_glr))