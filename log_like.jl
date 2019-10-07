#Following Hamilton(1994)
#TODO: Implement for omega =/= Identity

#data: lines are different dates, columns are different variables!

function loglike(PI,data)
    T = dim(data,1)
    n = dim(data,2)
    if(dim(PI,1) != n)
        @info "Dimensions do not match"
        return 99999999
    end
    bloc1 = -(T*n)/2*log(2*pi)
    aux1 = data[2,:]' - PI'*data[1,:]'
    sum_like = aux1'*aux1
    for j in 2:(T-1)
        aux1 = data[j+1,:]' - PI'*data[j,:]'
        sum_like = sum_like + aux1'*aux1
    end
    bloc3 = -1/2*sum_like
end
