# Calculating posterior probability from a mixture model

function gmmpdf(x, mix)

    prob = mix["priors"] .* calActiv(x,mix) 

    return sum(prob,2) 

end
