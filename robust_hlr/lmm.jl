function lmm(z,mix)

    # calculate the posterior
    post = aposterior(z, mix)

    # update means
    new_pr         = sum(post, 1)
    new_c          = post' * z
    mix["centres"] = new_c ./ new_pr'


    # update priors
    mix["priors"] = new_pr / sum(new_pr)

    # update covariance
    if mix["covarType"] == "diag"
        for i = 1:mix["ncentres"]
            diffs              = z .- mix["centres"][i,:] 
            mix["covars"][i,:] = sum(diffs .* diffs .* post[:,i], 1) ./ new_pr[i]
        end
    elseif mix["covarType"] == "cdiag"
        sums = 0
        for i = 1:mix["ncentres"]
            diffs = z .- mix["centres"][i,:] 
            sums  = sums + sum(diffs .* diffs .* post[:,i], 1) ./ new_pr[i]
        end
        for i = 1:mix["ncentres"]
            mix["covars"][i,:] = sums / mix["ncentres"]
        end
    elseif mix["covarType"] == "full"
        for i = 1:mix["ncentres"]
            diffs = z .- mix["centres"][i,:]
            # need sqrt() because we calculate (diffs * diffs) where
            # diffs is actually diffs * post,
            # so the result will be (diffs^2 * post^2)
            diffs                = diffs .* sqrt(post[:,i])
            mix["covars"][:,:,i] = (diffs' * diffs) / new_pr[i]
        end
    elseif mix["covarType"] == "cfull"
        sums = 0
        for i = 1:mix["ncentres"]
            diffs = z .- mix["centres"][i,:]
            diffs = diffs .* sqrt(post[:,i])
            sums  = sums + (diffs' * diffs) / new_pr[i]
        end
        for i = 1:mix["ncentres"]
            mix["covars"][:,:,i] = sums / mix["ncentres"]  + eye(mix["nin"])*1e-9
        end
    end

    return mix

end
