using LinearAlgebra

mutable struct svd_kalman
    A # state transition
    G #Observable transition
    Q  #observable variance
    R#variance of the state
    state_mean
    state_var
end

function start!(svd_kalman::svd_kalman,x0,var0)

    R = svd_kalman.R
    Q = svd_kalman.Q

    svd_kalman.state_mean = x0
    svd_var0 = svd(var0)
    svd_kalman.state_var = diagm(sqrt.(svd_var0.S))*svd_var0.Vt

    svd_R = svd(R)
    svd_kalman.R = diagm(sqrt.(svd_R.S))*svd_R.Vt

    svd_Q = svd(Q)
    svd_kalman.Q = diagm(sqrt.(svd_Q.S))*svd_Q.Vt

    return nothing
end

function update_state!(svd_kalman::svd_kalman,y;ll=false)

    R = svd_kalman.R
    old_var = svd_kalman.state_var
    G = svd_kalman.G
    old_state = svd_kalman.state_mean

    pre_gain = [R;old_var*G']
    svd_re = svd(pre_gain)

    K_bar = (old_var'*old_var)*G'*svd_re.V

    err = (y - G*old_state)
    err_bar = svd_re.Vt*err

    if size(old_state,1) == 1
        new_state = old_state + (K_bar*(svd_re.S).^(-2)*err_bar)[1]
    else
        new_state = old_state + K_bar*diagm((svd_re.S).^(-2))*err_bar
    end

    K = K_bar*diagm((svd_re.S).^(-2))*svd_re.Vt

    pre_var = [old_var*(I-K*G)';R*K']
    svd_var = svd(pre_var)

    new_var = diagm(svd_var.S)*svd_var.Vt

    svd_kalman.state_mean = new_state
    svd_kalman.state_var = new_var

    if ll == false
        return nothing
    else
        Re = diagm(svd_re.S)*svd_re.Vt
        return err, Re'*Re
    end
end

function forecast!(svd_kalman::svd_kalman)

    A = svd_kalman.A
    state_mean = svd_kalman.state_mean

    svd_kalman.state_mean = A*state_mean

    vari = svd_kalman.state_var
    Q = svd_kalman.Q

    pre_var = [vari*A';Q]
    svd_var = svd(pre_var)

    svd_kalman.state_var = diagm(svd_var.S)*svd_var.Vt

    return nothing

end

function update_and_forecast!(svd_kalman::svd_kalman,y;ll=false)
    if ll == false
        update_state!(svd_kalman,y)
        forecast!(svd_kalman)
        return nothing
    else
        μ,Σ = update_state!(svd_kalman,y,ll=true)
        forecast!(svd_kalman)
        return μ, Σ
    end
end

#loglikelihood
function loglikelihood(svd_kalman::svd_kalman,data)
    n = size(data,1)
    m = size(data,2)

    llh = zeros(n)

    G = svd_kalman.G
    R = svd_kalman.R
    μ = G*svd_kalman.state_mean
    Σ = G*svd_kalman.state_var*G' + R'*R

    for j in 1:n
        if m == 1
            llh[j] = -1/2*(m*log(2*pi) + logdet(Σ) + μ[1]^2/Σ[1])
        else
            llh[j] = -1/2*(m*log(2*pi) + logdet(Σ) + μ'*inv(Σ)*μ)
        end

        μ,Σ = update_and_forecast!(svd_kalman,data[j,:],ll=true)
    end
    return llh
end
