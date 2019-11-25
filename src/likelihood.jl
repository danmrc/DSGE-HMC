using QuantEcon
using Statistics
using LinearAlgebra

include(string(pwd(),"/src/gensys.jl"))

function log_like_dsge(par,data)
    #order to par
    #alfa
    #beta
    #epsilon
    #theta
    #sig
    #sigma this is the std dev of the innovation
    #phi_pi
    #phi_y
    #phi_v

    #data will have t x p dimension: lines are periods p are variables

    nobs = size(data,1)

    GAMMA_0 = [bet    0     0  0;
               1      sig   0  0;
               0      0     0  0;
               0      0     0  1]

    GAMMA_1 = [1      -kappa  0  0;
               0       sig    1  0;
               -phi_pi  -phi_y  1 -1;
               0       0      0  rho_v]

    PSI = [0; 0; 0; 1]

    PI = [bet  0;
          1    sig;
          0    0;
          0    0]

    p = size(GAMMA_1,1) #number of endogenous vars

    sol = gensys(GAMMA_0,GAMMA_1,PSI,PI)
    if sum(sol.eu) != 2
        return(-9999999999999)
    end

    Sig = zeros(p,p)
    Sig[4,4] = param[6]^2

    G = zeros(1,p)
    G[1,2] = 1

    A = sol.Theta1
    R = 0
    Q = sol.Theta2'*Sig
    kalman_res = Kalman(A,G,Q,R) #create a Kalman filter instance

    y_mean = mean(data,dims=1)
    y_var = var(data,dims=1)
    y_var = reshape(y_var,size(y_var,2))

    x_hat = repeat(y_mean,p) #initial mean of the state
    x_var = diagm(repeat(y_var,p))#variance initial of state
    set_state!(kalman_res,x_hat,x_var)

    inovs = zeros(nobs)

    llh = zero(nobs)

    for j in 1:nobs
        media = kalman_res.cur_x_hat
        varian = kalman_res.cur_sigma
        eta = data[j,:] - kalman_res.G*media #mean loglike
        P = kalman_res.G*varian*kalman_res.G' .+kalman_res.R#var loglike #TODO change this 
        llh[j] = -(p*log(2*pi) + logdet(P) .+ eta'*inv(P)*eta)[1]/2
        update!(kalman_res,data[j,:]) #updating the kalman estimates
    end
    return sum(llh)
end
