# Probability density function of the Gamma distribution

function gammapdf(x,k,theta)
    
    x = abs(x)
    y = ((x.^(k-1)).*exp(-x/theta))/(theta^k * factorial(k-1))

    return y
end

