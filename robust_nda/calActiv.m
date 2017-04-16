function lncc = calActiv(x, mix)

EPS = eps * 100000;

ndata = size(x, 1);
lncc  = zeros(ndata, mix.ncentres);  % Preallocate matrix

if (isempty(strfind(mix.covar_type, 'diag')))
    % if it does not contain 'diag' it is 'full' or 'cfull'
    % Ensure that no covariance is too small
    for j = 1:mix.ncentres
        %% comment 'while loop' out for speed
        %while min(svd(mix.covars(:,:,j))) < EPS
            mix.covars(:,:,j) = mix.covars(:,:,j) + (1e-1 * eye(mix.nin));
        %end
    end
    %mix.covars = mix.covars + (repmat(1e-7 * eye(mix.nin), [1 1 mix.ncentres]));
    
    if ndata/mix.nin < 10
        %% for KFLD, more stable
        normal =  mix.nin/2 * log(2*pi);
        
        for j = 1:mix.ncentres
            diffs = x - (ones(ndata, 1) * mix.centres(j, :));
            % Use Cholesky decomposition of covariance matrix to speed computation
            switch mix.covar_type
                case 'full'
                    [c, p] = chol(mix.covars(:,:,j));
                    if p
                        disp('Covariance is not positive definite')
                    end
                case 'cfull'
                    [c, p] = chol(mix.covars(:,:,1));
                    if p
                        disp('Covariance is not positive definite')
                    end
            end
            temp  = diffs/c;
            lncc(:, j) =  -0.5*sum(temp.*temp, 2) - normal - sum(diag(c));
        end
        %     %lncc = lncc - repmat(max(lncc, [], 2), 1, mix.ncentres);
    else
        % FOR rNDA, more stable
        normal = (2*pi)^(mix.nin/2);
        for j = 1:mix.ncentres
            diffs = x - (ones(ndata, 1) * mix.centres(j, :));
            % Use Cholesky decomposition of covariance matrix to speed computation
            c = chol(mix.covars(:, :, j));
            temp = diffs/c;
            lncc(:, j) = exp(-0.5*sum(temp.*temp, 2))./(normal*prod(diag(c)));
        end
    end
    
else
    % it contains 'diag', it is 'diag' or 'cdiag'
    % Ensure that no covariance is too small
    for j = 1:mix.ncentres
        index               = mix.covars(j,:) < EPS;
        mix.covars(j,index) = mix.covars(j,index) + EPS;
    end
    
    if ndata/mix.nin < 10        
        %% FOR KFLD
        normal = mix.nin/2*log(2*pi);
        s      = sum(.5*log(mix.covars), 2);
        for j = 1:mix.ncentres
            diffs      = x - (ones(ndata, 1) * mix.centres(j, :));
            lncc(:, j) = (-0.5*sum((diffs.*diffs)./(ones(ndata, 1) * ...
                mix.covars(j, :)), 2)) - normal - s(j);
        end
        %     %lncc = lncc - repmat(max(lncc, [], 2), 1, mix.ncentres);
    else
        %% FOR rNDA
        normal = (2*pi)^(mix.nin/2);
        s = prod(sqrt(mix.covars), 2);
        for j = 1:mix.ncentres
            diffs = x - (ones(ndata, 1) * mix.centres(j, :));
            lncc(:, j) = exp(-0.5*sum((diffs.*diffs)./(ones(ndata, 1) * ...
                mix.covars(j, :)), 2)) ./ (normal*s(j));
        end
    end

end
