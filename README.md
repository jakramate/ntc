# ntc
Label {N}oise {T}olerance Logistic Regression {C}lassifiers 

A library for training label-noise robust logistic regression written in Julia.


The library includes
- robust Logistic Regression (rLR) for class-dependent labal noise [1]
- generalised robust Logistic Regression (gLR) for instance-dependent label noise [2]

## Installation 
```
git clone https://www.github.com/jakramate/ntc
```

## Examples
Loading some dataset

```julia
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
tpnt = round(Int, 0.8 * dpnt)
```
Predefine results storage

```julia
err_rlr = zeros(repetitions,1)
err_glr = zeros(repetitions,1)
```

Main loop 
```
for i=1:repetitions

  # random train/test split
  perm = randperm(dpnt)
  xt   = x[perm[1:tpnt],:]
  yt   = y[perm[1:tpnt]]
  xs   = x[perm[tpnt+1:end],:]
  ys   = y[perm[tpnt+1:end]]

  xt, xs = standardise(xt, xs)

  yz, fd = injectLabelNoise(yt, [0.8 0.2;0.2 0.8])
```

Training the robust logistic regression
```julia
  # robust logistic regression
  w, g, llh = rlr(zeros(dim+1,1), [0.8 0.2;0.2 0.8], addbias(xt), yt, regType="lasso")
  err_rlr[i] = sum(sign(addbias(xs) * w) .!= castLabel(ys,-1))/length(ys)
```

Training the generalised robust logistic regression
```julia
  # generalised robust logistic regression
  wg, t0, t1, llhg = glr(zeros(dim+1,1), addbias(xt), yt, regType="lasso")
  err_glr[i] = sum(sign(addbias(xs) * wg) .!= castLabel(ys,-1))/length(ys)
```

Printing some stats 
```julia
  @printf("[Round %d] rlr = %.2f , glr = %.2f\n", i, err_rlr[i], err_glr[i])
  @printf("[log-likelihood] rlr = %.2f , glr = %.2f\n", llh[end], llhg[end])
  
end

@printf("[Mean] rlr = %.2f , glr = %.2f \n", mean(err_rlr), mean(err_glr))
```

## References
- [1] J. Bootkrajang, A. Kaban: Label-noise Robust Logistic Regression and its Applications, The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD 2012), 24-28 September 2012, Bristol, UK. 
- [2] J. Bootkrajang: A Generalised Label Noise Model for Classification in the Presence of Annotation Errors, Neurocomputing 192 (2016): 61-71 
