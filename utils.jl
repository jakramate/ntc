function addbias(x)
  return hcat(ones(size(x,1),1),x)
end
