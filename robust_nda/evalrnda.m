function [prob pred eIdx eRate] = evalrnda(x, y, mix, g)

% only for binary class
if length(unique(y))==2
    y = castLabel(y,2);
end

% ====================== BEGIN EVALUATING ESTIMATED MODEL ============================ 
TPNT   = size(x,1);
CLS    = mix.ncentres;

% %% FOR rKFD
% lncc   = calActiv(x, mix);
% lnpost = zeros(TPNT,CLS);
% 
% % calculate activation value
% for i = 1:CLS
%     flips       = repmat(g(:,i)', TPNT, 1);
%     priors      = repmat(mix.priors, TPNT, 1);
%     lnpost(:,i) = lncc(:,i) + log(sum(flips .* priors, 2));    
% end
% 
% lnpost = lnpost - repmat(max(lnpost, [], 2), 1, mix.ncentres);
% 
% post   = exp(lnpost);
% denom  = sum(post,2);
% denom  = denom + (denom == 0);
% post   = post ./ (denom * ones(1, CLS));

% FOR rNDA
acct = calActiv(x, mix);
post = zeros(TPNT,CLS);

% calculate activation value
for i = 1:CLS
    flips  = repmat(sum(g(:,i)), TPNT, 1);
    priors = repmat(mix.priors(i), TPNT, 1);
    num    = acct(:,i) .* flips .* priors;
    denom  = sum(acct,2) .* flips .* priors;
    denom  = denom + (denom == 0);
    post(:,i) = num ./ denom;
end

% get the prediction from 'max' which returns index of maximum value in each col
[prob pred] = max(post, [], 2); 
eIdx        = (pred == y);
eRate       = sum(eIdx)/TPNT;