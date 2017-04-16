% calculate posterior probability of regular mixture model
function [post lncc flips] = calPost(x, y, mix, g)

ndata = size(x, 1);
lncc = calActiv(x, mix);

flips = zeros(size(x,1), mix.ncentres);
prior = zeros(size(x,1), mix.ncentres);

for i = 1:mix.ncentres
    flips(y==i,:) = repmat(g(i,:), sum(y==i,1), 1);
    prior(y==i,:) = repmat(mix.priors(i), sum(y==i,1), mix.ncentres);
end


if ndata/mix.nin < 10
    %% FOR KFLD
    lnpost = lncc + log(flips) + log(prior);
    lnpost = lnpost - repmat(max(lnpost, [], 2), 1, mix.ncentres);
    post   = exp(lnpost);
    denom  = sum(post,2);
    denom  = denom + (denom == 0);
    post   = post ./ (denom * ones(1, mix.ncentres));
else
    % FOR rNDA
    post  = lncc .* flips .* prior;     % numerator
    denom = sum(post,2);                % denominator
    denom = denom + (denom == 0);
    post  = post ./ (denom * ones(1, mix.ncentres));
end
