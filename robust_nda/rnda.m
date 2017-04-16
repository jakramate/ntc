% Name  :  Fisher Discriminant with noise model
% Author:  Jakramate Bootkrajang
% input :  bool NM = using noise modelling ?
%          bool VB = shows addional info (verbose)?
%          The rest are training/test x/y
% return:  Generalisation error 

function [mix g llh] = rnda(mix, g, x, y, options)

% initialise parameters
for i=1:mix.ncentres
    
    mix.centres(i, :) = mean(x(y == i, :));
    %mix.centres(i, :) = rand(1,mix.nin);
    
    switch mix.covar_type
        case 'diag'
            mix.covars(i, :)    = diag(cov(x(y == i, :)))';
        case 'cdiag'
            mix.covars(i, :)    = diag(cov(x))' + rand(1,mix.nin);
        case 'full'
            mix.covars(:, :, i) = cov(x(y == i, :));
        case 'cfull'
            mix.covars(:, :, i) = cov(x) + eye(mix.nin)*1e-9;
    end
    
    if options.map
        mix.priors(i) = size(x(y == i),1) / size(x,1);
    else
        mix.priors(i) = 1/mix.ncentres;
    end
end

% learning the model
for l=1:options.maxIter
    
    % calculate the posterior
    [post lncc flips] = calPost(x, y, mix, g);

    % update means
    new_pr      = sum(post, 1);
    new_c       = post' * x;    
    mix.centres = new_c ./ (new_pr' * ones(1, mix.nin));
     
    % update priors
    if options.map
        mix.priors = new_pr/sum(new_pr);
    end
    
    % update covariance
    switch mix.covar_type
        case 'diag'
            for i = 1:mix.ncentres
                diffs           = x - (ones(size(x,1), 1) * mix.centres(i,:));
                mix.covars(i,:) = sum((diffs .* diffs) .* (post(:,i) * ones(1, ...
                mix.nin)), 1) ./ new_pr(i);
            end
        case 'cdiag'
            sums = 0;
            for i = 1:mix.ncentres
                diffs = x - (ones(size(x,1), 1) * mix.centres(i,:));
                sums  = sums + sum((diffs .* diffs) .* (post(:,i) * ones(1, ...
                mix.nin)), 1) ./ new_pr(i);
            end
            for i = 1:mix.ncentres
                mix.covars(i,:) = sums / mix.ncentres;
            end           
        case 'full'
            for i = 1:mix.ncentres
                diffs = x - (ones(size(x,1), 1) * mix.centres(i,:));
                % need sqrt() because we calculate (diffs * diffs) where
                % diffs is actually diffs * post, 
                % so the result will be (diffs^2 * post^2)
                diffs             = diffs .* (sqrt(post(:,i)) * ones(1, mix.nin)); 
                mix.covars(:,:,i) = (diffs' * diffs) / new_pr(i);                          
%               mix.covars(:,:,i) = mtimesx(diffs,'T', diffs) / new_pr(i);                          
            end
        case 'cfull'
            sums = 0;
            for i = 1:mix.ncentres
                diffs = x - (ones(size(x,1), 1) * mix.centres(i,:));
                diffs = diffs .* (sqrt(post(:,i)) * ones(1, mix.nin)); 
                sums  = sums + (diffs' * diffs) / new_pr(i);                
            end            
            for i = 1:mix.ncentres
                mix.covars(:,:,i) = sums / mix.ncentres  + eye(mix.nin)*1e-9;
            end        
    end % end switch    

    
    % update gamma table
    if options.estG
        for j = 1:mix.ncentres
            index = (y == j);
            num   = sum(post(index,:),1);
            denom = sum(num,2);
            for k = 1:mix.ncentres
                g(j,k) = num(k)/denom;
            end
        end
    end   
    
    % log likelihood 
    llh(l) = sum(sum(exp(lncc) + flips, 2));

end % end optimisation loop




