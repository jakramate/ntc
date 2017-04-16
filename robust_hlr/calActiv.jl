# calculating gmm activation value

function calActiv(x, mix)

    ndata = size(x, 1)
    lncc  = zeros(ndata, mix["ncentres"])  # Preallocate matrix

    if contains(mix["covarType"], "full")
        # if it does not contain 'diag' it is 'full' or 'cfull'
        # Ensure that no covariance is too small
        for j = 1:mix["ncentres"]
            # rectify sigular covariance matrix
            while true
                U,S,V = svd(mix["covars"][:,:,j])
                minimum(S) > 1e-8 && break   
                mix["covars"][:,:,j] = mix["covars"][:,:,j] + (1e-8 * eye(mix["nin"]))
            end
        end
    
        if ndata/mix["nin"] < 10
            # this version is more stable when dim is >> than ndata
            normal =  mix["nin"]/2 * log(2*pi)
        
            for j = 1:mix["ncentres"]
                diffs = x .- mix["centres"][j, :]
                # Use Cholesky decomposition of covariance matrix to speed computation
                if mix["covarType"] == "full"
                        c = chol(mix["covars"][:,:,j])
                elseif mix["covarType"] == "cfull"
                        c = chol(mix["covars"][:,:,1])
                end
                temp  = diffs/c
                lncc[:, j] =  -0.5*sum(temp.*temp, 2) - normal - sum(diag(c))
            end
        else
            # this is more stable when we have enough data points
            normal = (2*pi).^(mix["nin"]/2)
            for j = 1:mix["ncentres"]
                diffs = x .- mix["centres"][j, :]
                # Use Cholesky decomposition of covariance matrix to speed computation
                c     = chol(mix["covars"][:, :, j])
                temp  = diffs/c
                lncc[:, j] = exp(-0.5*sum(temp.*temp, 2))./(normal*prod(diag(c)))
            end
        end    
    else
        # it does not contains 'full' so it must either be 'diag' or 'cdiag'
        # Ensure that no covariance is too small
        for j = 1:mix["ncentres"]
          #  index                  = mix["covars"][j,:] .< 1e-8
          #  println(index)
          #  mix["covars"][j,index] = mix["covars"][j,index] + 1e-8
          mix["covars"][j,:] = map(rectify, mix["covars"][j,:])
        end
    
        if ndata/mix["nin"] < 10        
            # this version is more stable when dim is >> than ndata
            normal = mix["nin"]/2*log(2*pi)
            s      = sum(0.5*log(mix["covars"]), 2)
            for j = 1:mix["ncentres"]
                diffs      = x .- mix["centres"][j, :]
                lncc[:, j] = (-0.5*sum((diffs.*diffs)./ mix["covars"][j, :], 2)) - normal - s[j]
            end
        else
            # this is more stable when we have enough data points
            normal = (2*pi).^(mix["nin"]/2)
            s = prod(sqrt(mix["covars"]), 2)
            for j = 1:mix["ncentres"]
                diffs      = x .- mix["centres"][j, :]
                lncc[:, j] = exp(-0.5*sum((diffs.*diffs) ./ mix["covars"][j, :], 2)) ./ (normal*s[j])
            end
        end

    end

    return lncc
end

# function for rectifying too small covariance element
function rectify(x)
    if x < 1e-8
        x = x + 1e-8
    end
    
    return x
end
