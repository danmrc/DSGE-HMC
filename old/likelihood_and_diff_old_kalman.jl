using QuantEcon
using Statistics
using LinearAlgebra

include(string(pwd(),"/src/gensys.jl"))
include(string(pwd(),"/hmc/diffs_estrut_v3.jl"))
include(string(pwd(),"/hmc/diff_kalman.jl"))

function solve_lyapunov_vec(A,Q)
    aux = (I - kron(A,A))
    vec_ans = \(aux,vec(Q))
    res = reshape(vec_ans,size(Q))
    return res
end


function log_like_dsge(par,data;kalman_tol = 1e-10,reduce=true)
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
    l = size(data,2)

    GAMMA_0 = [bet  0    0  0;
               1    sig  0  0;
               0    0    0  0;
               0    0    0  1]

    GAMMA_1 = [ 1       -kappa  0   0;
                0        sig    1   0;
               -phi_pi  -phi_y  1  -1;
                0        0      0   rho_v]

    PSI = [0; 0; 0; par[6]]

    PI = [bet  0;
          1    sig;
          0    0;
          0    0]

    p = size(GAMMA_1,1) #number of endogenous vars

    sol = gensys(GAMMA_0,GAMMA_1,PSI,PI; verbose = false)

    if sum(sol.eu) != 2
        return -Inf, repeat([0],length(par))
    end

    #Sig = zeros(p,p)
    #Sig[4,4] = par[6]

    G = zeros(1,p)
    G[1,2] = 1

    A = sol.Theta1
    A[abs.(A) .< 10*eps()] .= 0
    R = [0] .+ 1e-2
    Q = sol.Theta2*sol.Theta2'
    Q_eig = eigen(Q)
    #Q = Q_eig.vectors*diagm(Q_eig.values .+ mean(Q_eig.values))*Q_eig.vectors'

    kalman_res = Kalman(A,G,Q,R) #create a Kalman filter instance

    y_mean = mean(data,dims=1)
    y_var = var(data,dims=1)
    y_var = reshape(y_var,size(y_var,2))
    y_var = repeat([1],p)

    x_hat = repeat(y_mean,p) #initial mean of the state
    x_var = I(4) #diagm(repeat(y_var,p))#variance initial of state
    #x_var = x_var*x_var'
    set_state!(kalman_res,x_hat,x_var)

    fit = zeros(nobs,4)

    dA,dB = diff_mod(par)
    #dB = dB[1:4,:]
    dA = dA'*I
    dG = zeros(10,4)
    dQ = (kron(sol.Theta2,I(p)) + kron(I(p),sol.Theta2)*commutation_matrix(4,1))*dB
    dQ = dQ'
    dR = zeros(size(par,1),1)

    llh = zeros(nobs)
    dll = zeros(nobs,size(par,1))

    dx0 = zeros(p)
    S0 = zeros(p,p)

    g_x_hat_l = zeros(size(par,1),p)
    P = kalman_res.G*S0*kalman_res.G' + kalman_res.R
    d_Sl = transpose(diff_S0(A,S0,dA,dQ)) #vec, not vech

    for j in 1:nobs
        med = kalman_res.cur_x_hat
        fit[j,:] = med
        varian = kalman_res.cur_sigma
        #println(det(varian))
        eta = data[j,:] - kalman_res.G*med #mean loglike
        P = kalman_res.G*varian*kalman_res.G' + kalman_res.R
        teste_cond = 1/cond(P)

        gain = (A*varian*G')*inv(P)

        d_p = transpose(diff_P(G,varian,dG,d_Sl,dR))
        d_gain = transpose(diff_K(A,P,G,varian,dA,dG,d_p,d_Sl))
        g_eta = transpose(grad_y(med,g_x_hat_l,G,dG))
        g_x_hat = transpose(grad_x_hat(med,dA,A,g_x_hat_l,eta,g_eta, gain,d_gain))

        p_inv = inv(P)

        if teste_cond < kalman_tol
            llh[j] = -500
            dll[j,:] .= 0
        #elseif det(P) <= 0
        #    llh[j] = -500
        else
            llh[j] = -(p*log(2*pi) + logdet(P) .+ eta'*p_inv*eta)/2
            dll[j,:] = -1/2*p_inv*d_p' - (eta'*p_inv)*g_eta' + 1/2*kron(eta'*p_inv,eta'*p_inv)*d_p'
            QuantEcon.update!(kalman_res,data[j,:]) #updating the kalman estimates
        end #end if
        #println(kalman_res.cur_sigma)
        #println(det(varian))
        #println(P)
        g_x_hat_l = g_x_hat
        d_Sl = transpose(diff_S(A,P,gain,varian,dA,d_p,d_gain,d_Sl,dQ))
    end #end for
    llh = llh[10:length(llh)]
    dll = dll[10:length(llh),:]
    if reduce == true
        return sum(llh),sum(dll,dims=1)
    else
        return llh,dll
    end
end
