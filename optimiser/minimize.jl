function minimize(X, f, nFeval, varargin...)


# X is a column vector
# f must also return column vector of partial derivatives


# Minimize a differentiable multivariate function. 
#
# Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
#
# where the starting point is given by "X" (D by 1), and the function named in
# the string "f", must return a function value and a vector of partial
# derivatives of f wrt X, the "nFeval" gives the length of the run: if it is
# positive, it gives the maximum number of line searches, if negative its
# absolute gives the maximum allowed number of function evaluations. 
# Unlike the original minimize.m the second component is not yet supported. 
# 
# The parameters P1, P2, P3, ... are passed on to the function f.
#
# The function returns when either its length is up, or if no further progress
# can be made (ie, we are at a (local) minimum, or so close that due to
# numerical problems, we cannot get any closer). NOTE: If the function
# terminates within a few iterations, it could be an indication that the
# function values and derivatives are not consistent (ie, there may be a bug in
# the implementation of your "f" function). The function returns the found
# solution "X", a vector of function values "fX" indicating the progress made
# and "i" the number of iterations (line searches or function evaluations,
# depending on the sign of "length") used.
#
# The Polack-Ribiere flavour of conjugate gradients is used to compute search
# directions, and a line search using quadratic and cubic polynomial
# approximations and the Wolfe-Powell stopping criteria is used together with
# the slope ratio method for guessing initial step sizes. Additionally a bunch
# of checks are made to make sure that exploration is taking place and that
# extrapolation will not be unboundedly large.
#
# See also: checkgrad 
#
# Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

INT = 0.1    # don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0                  # extrapolate maximum 3 times the current step-size
MAX = 20                         # max 20 function evaluations per line search
RATIO = 10                                       # maximum allowed slope ratio
SIG = 0.1 
RHO = SIG/2 

# SIG and RHO are the constants controlling the Wolfe-
# Powell conditions. SIG is the maximum allowed absolute ratio between
# previous and new slopes (derivatives in the search direction), thus setting
# SIG to low (positive) values forces higher precision in the line-searches.
# RHO is the minimum allowed fraction of the expected (from the slope at the
# initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
# Tuning of SIG (depending on the nature of the function to be optimized) may
# speed up the minimization it is probably not worth playing much with RHO.

# The code falls naturally into 3 parts, after the initial line search is
# started in the direction of steepest descent. 1) we first enter a while loop
# which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
# have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
# enter the second loop which takes p2, p3 and p4 chooses the subinterval
# containing a (local) minimum, and interpolates it, unil an acceptable point
# is found (Wolfe-Powell conditions). Note, that points are always maintained
# in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
# conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
# was a problem in the previous line-search. Return the best value so far, if
# two consecutive line-searches fail, or whenever we run out of function
# evaluations or line-searches. During extrapolation, the "f" function may fail
# either with an error or returning Nan or Inf, and minimize should handle this
# gracefully.
#
#
# Ported to Julia by Jakramate Bootkrajang (2015)
#

# initializing all variables

f0,f1,f2,f3,f4 = 0,0,0,0,0
x0,x1,x2,x3,x4 = 0,0,0,0,0
d0,d1,d2,d3,d4 = 0,0,0,0,0
df0 = zeros(length(X),1)
df1 = zeros(length(X),1)
df2 = zeros(length(X),1)
df3 = zeros(length(X),1)
df4 = zeros(length(X),1)

#if maximum(size(len_in)) == 2
#    red = len_in[2]
#    len = len_in[1]
#else 
    red = 1
    len = nFeval
#end


if len > 0
    S = "Linesearch" 
else 
    S = "Function evaluation" 
end 

i = 0                                         # zero the run len counter
ls_failed = false                             # no previous line search has failed
f0, df0 = f(X, varargin...)                   # get function value and gradient
fX = f0
i = i + convert(Int64,(len<0))                # count epochs?!
s = -df0; d0 = (-s'*s)[1]                     # initial search direction (steepest) and slope
x3 = red/(1-d0)                               # initial step is red/(|s|+1)

while i < abs(len)                            # while not finished
  
  i = i + convert(Int64,(len>0))              # count iterations?!
  X0 = X; F0 = f0; dF0 = df0                  # make a copy of current values
  
  if len > 0
    M = MAX 
  else 
    M = minimum([MAX, -len-i]) 
  end

  while true                             # keep extrapolating as long as necessary
    x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0
    succeed = false 

    while ~succeed && (M > 0)
      try
        M = M - 1 
        i = i + convert(Int64,(len<0))                         # count epochs?!
        

        f3, df3 = f(X+x3*s, varargin...)


        #println(any(isnan(df3) | isinf(df3)))
        if isnan(f3) || isinf(f3) || any(isnan(df3)|isinf(df3))
            error("NaN or Inf encountered")
        end
        succeed = true
        

      catch                                # catch any error which occured in f
        x3 = (x2+x3)/2                                  # bisect and try again
      end
    end
    
    if f3 < F0 
        X0  = X+x3*s; F0  = f3; dF0 = df3
    end                                                     # keep best values

    d3 = (df3'*s)[1]     # only for Julia where 1x1Array is not a number,  # new slope

   # println(df3, " ", d0)

    if (d3 > SIG*d0) || (f3 > f0+x3*RHO*d0) || (M == 0)  # are we done extrapolating?
      break
    end
    x1 = x2; f1 = f2; d1 = d2                        # move point 2 to point 1
    x2 = x3; f2 = f3; d2 = d3                        # move point 3 to point 2
    
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)                  # make cubic extrapolation
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)

    cx3 = x1-d1*(x2-x1)^2/(B+sqrt(complex(B*B-A*d1*(x2-x1)))) # num. error possible, ok!
    # because of type promotion cx3 is now complex
    # so we extract its real part and save it to 'x3'
    x3 = real(cx3)

    if ~isreal(cx3) || isnan(x3) || isinf(x3) || (x3 < 0)  # num prob | wrong sign?
      x3 = x2*EXT                                     # extrapolate maximum amount
    elseif x3 > x2*EXT                     # new point beyond extrapolation limit?
      x3 = x2*EXT                                     # extrapolate maximum amount
    elseif x3 < x2+INT*(x2-x1)            # new point too close to previous point?
      x3 = x2+INT*(x2-x1)
    end
  end                                                          # end extrapolation


  while ((abs(d3) > (-SIG*d0)) || (f3 > f0+x3*RHO*d0)) && (M > 0)  # keep interpolating
  

    if (d3 > 0) || (f3 > f0+x3*RHO*d0)                 # choose subinterval
      x4 = x3; f4 = f3; d4 = d3                        # move point 3 to point 4
    else
      x2 = x3; f2 = f3; d2 = d3                        # move point 3 to point 2
    end
    
    if f4 > f0           
      x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2))    # quadratic interpolation
    else
      A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)                  # cubic interpolation
      B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
      x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A           # num. error possible, ok!
    end
    
    if isnan(x3) || isinf(x3)
      x3 = (x2+x4)/2                  # if we had a numerical problem then bisect
    end
    x3      = maximum([minimum([x3, x4-INT*(x4-x2)]),x2+INT*(x4-x2)])  # don't accept too close
    f3, df3 = f(X+x3*s, varargin...)
    if f3 < F0 
        X0  = X+x3*s; F0  = f3; dF0 = df3 
    end                                                         # keep best values

    M = M - 1 
    i = i + convert(Int64,(len<0))                             # count epochs?!
    d3 = (df3'*s)[1]                                              # new slope

  end                                                       # end interpolation

  if (abs(d3) < (-SIG*d0)) && (f3 < (f0+x3*RHO*d0))          # if line search succeeded
    X = X+x3*s; f0 = f3; fX = [fX' f0]'                     # update variables
    #@printf("%s %d;  Value %4.6f\n", S, i, f0)         # add verbosity here
   
    s   = (df3'*df3-df0'*df3)/(df0'*df0).*s - df3   # Polack-Ribiere CG direction
    df0 = df3                                               # swap derivatives
    d3  = d0; d0  = (df0'*s)[1]
    
    if d0 > 0                                      # new slope must be negative
      s  = -df0; d0 = (-s'*s)[1]                  # otherwise use steepest direction
    end
    
    x3 = x3 * minimum([RATIO, d3/(d0-realmin())])  # slope ratio but max RATIO
    ls_failed = false                              # this line search did not fail
  else
    X   = X0; f0  = F0; df0 = dF0                     # restore best point so far
   

    if ls_failed || (i > abs(len))         # line search failed twice in a row
      break                             # or we ran out of time, so we give up
    end
    
    s  = -df0; d0 = (-s'*s)[1]    #only for Julia where 1x1Array is not a number 
                                                                    # try steepest
    x3 = 1/(1-d0)                     
    ls_failed = true                                    # this line search failed
  end
end

#println()  #add verbosity here

return X, fX, i

end
