function solve_lyapunov(A,Q,guess = I;iter_lim=100,epsilon=1e-14)
    iter = 1
    err = 1
    while iter <= iter_lim && err > epsilon
        new_guess = A*guess*A' + Q
        err = maximum(abs.(new_guess - guess))
        guess = new_guess
        iter += 1
    end
    return(guess)
end

using LinearAlgebra

function solve_lyapunov_vec(A,Q)
    aux = (I - kron(A,A))
    vec_ans = \(aux,vec(Q))
    res = reshape(vec_ans,size(Q))
    return res
end

using ToeplitzMatrices

function build_variance(g,h,Q,T) #only works for a single series
    sig_x = solve_lyapunov_vec(h,Q)
    y_autocov(t) = g*h*sig_x*h'^(t)*g'
    vec_st = map(x->first(y_autocov(x)),1:(T))
    res = Toeplitz(vec_st,vec_st)
    return res
end
