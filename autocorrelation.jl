using Statistics

function autocor(x,order::Int64)
    n = size(x,1)
    lag_x = x[1:(n-order),:]
    xx = x[(order+1):n,:]
    return cor(xx,lag_x)
end

function autocor(x,order::UnitRange{Int64})
    res = zeros((size(x,2),size(x,2),length(order)))
    for j in order
        res[:,:,j] = autocor(x,j)
    end
    return res
end

# u = randn(1000)
#
# y = zeros(1000)
#
# for t = 1:999
#     y[t+1] = 0.5y[t] + u[t]
# end
