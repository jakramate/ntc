# calculate posterior probability of regular mixture model
# mix is a Dict representing mixture model's data
function aposterior(x, mix)

ndata = size(x, 1)
lncc  = calActiv(x, mix)
prior = ones(size(x,1), mix["ncentres"])

for i = 1:mix["ncentres"]
    prior[:,i] = prior[:,i] * mix["priors"][i]
end


if ndata/mix["nin"] < 10
    # FOR KFLD, using log of things to avoid numerical problems
    lnpost = lncc + log(prior)
    lnpost = broadcast(-, lnpost, maximum(lnpost, 2))
    post   = exp(lnpost)
    denom  = sum(post,2)
    denom  = denom + (denom .== 0)
    post   = post ./ (denom * ones(1, mix["ncentres"]))
else
    # FOR rNDA
    post  = lncc .* prior     # numerator
    denom = sum(post,2)               # denominator
    denom = denom + (denom == 0)
    post  = post ./ (denom * ones(1, mix["ncentres"]))
end

return post

end
