using QuantEcon
using Statistics
using LinearAlgebra

include(string(pwd(),"/src/gensys.jl"))

function log_like_dsge(par,data;kalman_tol = 1e-10)
    #order to par
    #alfa
    #beta
    #epsilon
    #theta
    #sig
    #sigma: this is the std dev of the innovation
    #phi
    #phi_pi
    #phi_y
    #rho_v

    #data will have t x p dimension: lines are periods p are variables

    alfa = par[1]
    bet = par[2]
    epsilon = par[3]
    theta = par[4]
    sig = par[5]
    #par 6 is coded bellow see line 56
    phi = par[7]
    phi_pi = par[8]
    phi_y = par[9]
    rho_v = par[10]

    THETA = (1-alfa)/(1-alfa+alfa*epsilon)
    lamb = (1-theta)*(1-bet*theta)/theta*THETA
    kappa = lamb*(sig+(phi+alfa)/(1-alfa))

    nobs = size(data,1)

    GAMMA_0 = [bet  0    0  0;
               1    sig  0  0;
               0    0    0  0;
               0    0    0  1]

    GAMMA_1 = [ 1       -kappa  0   0;
                0        sig    1   0;
               -phi_pi  -phi_y  1  -1;
                0        0      0   rho_v]

    PSI = [0; 0; 0; 1]

    PI = [bet  0;
          1    sig;
          0    0;
          0    0]

    p = size(GAMMA_1,1) #number of endogenous vars

    sol = gensys(GAMMA_0,GAMMA_1,PSI,PI; verbose = false)
    if sum(sol.eu) != 2
        return -99999999999
    end

    #Sig = zeros(p,p)
    #Sig[4,4] = par[6]

    G = zeros(1,p)
    G[1,2] = 1

    A = sol.Theta1
    R = [0] .+ 1e-8
    Q = par[6]^2*sol.Theta2*sol.Theta2'

    kalman_res = Kalman(A,G,Q,R) #create a Kalman filter instance

    y_mean = mean(data,dims=1)
    y_var = var(data,dims=1)
    y_var = reshape(y_var,size(y_var,2))
    #y_var = repeat([1],p)

    x_hat = repeat(y_mean,p) #initial mean of the state
    x_var = diagm(repeat(y_var,p))#variance initial of state
    #x_var = x_var*x_var'
    set_state!(kalman_res,x_hat,x_var)

    fit = zeros(nobs,4)

    llh = zeros(nobs)

    for j in 1:(nobs-1)
        med = kalman_res.cur_x_hat
        fit[j,:] = med
        varian = kalman_res.cur_sigma
        #println(det(varian))
        eta = data[j+1,:] - kalman_res.G*med #mean loglike
        P = kalman_res.G*varian*kalman_res.G' + kalman_res.R
        teste_cond = 1/cond(P)
        if teste_cond < kalman_tol
            llh[j] = -500
        #elseif det(P) <= 0
        #    llh[j] = -500
        else
            llh[j] = -(p*log(2*pi) + logdet(P) .+ eta'*inv(P)*eta)/2
            QuantEcon.update!(kalman_res,data[j+1,:]) #updating the kalman estimates
        end
        #println(kalman_res.cur_sigma)
        #println(det(varian))
        #println(P)
    end
    llh = llh[10:length(llh)]
    return sum(llh)
end
