# Description
#        Function to cast between different type of
#        label representation i.e. from {-1,1} to {0,1} to {1,2}
# Author
#        Jakramate Bootkrajang
# Input
#        y = label vector
#        t = target representation choose from {-1,0,2}
# Output
#        y = casted label vector
# ==========================================================================

function castLabel(y, t)

  if -1 in y
    if t == -1
      y_new = y;
    elseif t == 0
      y_new = (y + 1) ./ 2
    elseif t == 2
      y_new = (y + 3) ./ 2
    end
  elseif 0 in y
    if (t == -1)
      y_new = y .* 2 - 1
    elseif (t == 0)
      y_new = y
    elseif (t == 2)
      y_new = y + 1
    end
  elseif 2 in y
    if (t == -1)
      y_new = y .* 2 - 3
    elseif (t == 0)
      y_new = y - 1
    elseif (t == 2)
      y_new = y
    end
  end

  return y_new
end