using Distributions, LinearAlgebra,Parameters

struct rkhs_reg
    alf
    sig
    x
end

function rkhs_gaussian(x::Array{<:Any,1},y,sig,lambda)
    N = size(x,1)
    #check dimensions
    if size(y,1) != N
        @error "Dimension Mismatch"
    end
    kernel_mat = zeros(N,N)
    kernel(x,z) = pdf(Normal(z,sig),x)
    for i in 1:N
        kernel_mat[:,i] = [kernel(x[j],x[i])/N for j in 1:N]
    end
    alf = pinv(kernel_mat + lambda*I(N))*y/sqrt(N)
    return rkhs_reg(alf,sig,x)
end

function rkhs_gaussian(x::Array{<:Any,2},y,sig,lambda)
    N = size(x,1)
    #check dimensions
    if size(y,1) != N
        @error "Dimension Mismatch"
    end
    kernel_mat = zeros(N,N)
    kernel(x,z) = pdf(MvNormal(z,sig),x)
    for i in 1:N
        kernel_mat[:,i] = [kernel(x[j,:],x[i,:])/N for j in 1:N]
    end
    alf = pinv(kernel_mat + lambda*I(N))*y/sqrt(N)
    return rkhs_reg(alf,sig,x)
end

function(model::rkhs_reg)(val)
    @unpack alf,sig,x = model
    N = size(x,1)
    kern_val = Array{Any}(undef,N)
    if size(x,2) == 1
        for i in 1:N
            kern_val[i] = pdf(Normal(x[i],sig),val)
        end
    else
        for i in 1:N
            kern_val[i] = pdf(MvNormal(x[i,:],sig),val)
        end
    end
    return 1/sqrt(N)*alf'*kern_val
end
