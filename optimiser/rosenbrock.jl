function rosenbrock(x, args...)

    D = length(x)
    f = sum(100*(x[2:D]-x[1:D-1].^2).^2 + (1-x[1:D-1]).^2)

    df = zeros(D, 1);
    df[1:D-1] = - 400*x[1:D-1].*(x[2:D]-x[1:D-1].^2) - 2*(1-x[1:D-1]);
    df[2:D] = df[2:D] + 200*(x[2:D]-x[1:D-1].^2);


    return f, df

end
