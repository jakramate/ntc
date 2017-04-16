# Name  : A function to inject label noise
# Author: Jakramate Bootkrajang
# Input : y           = true label {1,...n} representation
#         targetClass = {0,1,2....} s.t. 0 is symmetric NoiseRateing
#         NoiseRate   = (0,1)
# Output: yz          = noisy label  {1,...n} representation
#         fp          = flip indicator vector


function injectLabelNoise(y, flipRate)

    fd  =  -ones(size(y))
    yz  =  castLabel(y,-1)
    y   =  castLabel(y,2)

    # sampling some numbers
    for i=1:2
        prob    = rand(size(y))   
        idx     = find((y.==i) & (prob .<= flipRate[i]))
        yz[idx] = -yz[idx]
        fd[idx] = -fd[idx]
    end

    yz = castLabel(yz,2)

    return yz, fd

end
