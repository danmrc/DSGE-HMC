using QuantEcon
using Statistics
using LinearAlgebra

include(string(pwd(),"/src/gensys.jl"))

function log_like_dsge(par,data;kalman_tol = 1e-10)
    #include(string(pwd(),"/matrices_carlos_giba.jl"))

    #par = [0.25 0.989 1 # alfa, beta, phi
    #       0.6 -0.2 -0.3 # A0
    #       0.6 0.1 -0.3 0.1 0.5 -0.2 0.1 0.8 0.5 # A1
    #       0.3 0.5 0.6 # shocks_star
    #       0.68 1.27 0.58 0.07 0.82 0.13 0.92 0.02 0.79 0.60 0.12 0.88 5.57
    #      10.34 2.08 1.10 0.54 0.30 1.99 0.70 0.31 0.79] # estimated parameters
           #29 dec: Calibrated the EUA VAR for the data we have

    # calibrated parameters
    alfa    = par[1]
    bet     = par[2]
    phi     = par[3]
    a_0piy  = par[4]
    a_0iy   = par[5]
    a_0ipi  = par[6]
    a_1yy   = par[7]
    a_1ypi  = par[8]
    a_1yi   = par[9]
    a_1piy  = par[10]
    a_1pipi = par[11]
    a_1pii  = par[12]
    a_1iy   = par[13]
    a_1ipi  = par[14]
    a_1ii   = par[15]
    sigma_y_star  = par[16]
    sigma_pi_star = par[17]
    sigma_i_star  = par[18]

    # estimated parameters
    h           = par[19]
    sig         = par[20]
    eta         = par[21]
    delta_D     = par[22]
    theta_D     = par[23]
    delta_I     = par[24]
    theta_I     = par[25]
    chi         = par[26]
    rho_a       = par[27]
    rho_gamma   = par[28]
    rho_cp      = par[29]
    rho_phi     = par[30]
    sigma_a     = par[31]
    sigma_gamma = par[32]
    sigma_cp    = par[33]
    sigma_phi   = par[34]
    rho_1       = par[35]
    rho_2       = par[36]
    lambda_pi   = par[37]
    lambda_y    = par[38]
    lambda_s    = par[39]
    sigma_i     = par[40]

    G0 = zeros(24,24)
    G1 = zeros(24,24)
    Psi = zeros(24,8)
    Pi = zeros(24,6)

    # Equation 1 (Euler)

    G0[1,1] = -1
    G0[1,2] = -(1-h)/sig
    G0[1,3] = (1-h)/sig
    G0[1,7] = 1+h
    G0[1,8] = (1-h)/sig

    G1[1,3] = (1+h)/sig
    G1[1,7] = h

    Pi[1,1] = -1
    Pi[1,2] = -(1-h)/sig
    Pi[1,3] = (1-h)/sig

    # Equation 2 (goods market equilibrium)

    G0[2,9]  = 1
    G0[2,7]  = -(1-alfa)
    G0[2,10] = -alfa*eta*(2-alfa)
    G0[2,11] = -alfa*eta
    G0[2,17] = -alfa

    # Equation 3 (tot)

    G0[3,10] = 1
    G0[3,13] = -1
    G0[3,12] = 1

    G1[3,10] = 1

    # Equation 4 (relationship q and tot)

    G0[4,10] = -(1-alfa)
    G0[4,14] = 1
    G0[4,11] = -1

    # Equation 5 (relationship q, s, pi)

    G0[5,14] = 1
    G0[5,15] = -1
    G0[5,18] = -1

    G1[5,2]  = -1
    G1[5,14] = 1
    G1[5,15] = -1

    # Equation 6 (phillips curve - domestic)

    G0[6,5]  = -bet
    G0[6,12] = 1+bet*delta_D
    G0[6,24] = -bet*(1-theta_D)*(1-theta_D*bet)/theta_D

    G1[6,12] = delta_D

    Pi[6,4] = -bet

    # Equation 7 (marginal cost)

    G0[7,24] = 1
    G0[7,10] = -alfa
    G0[7,9]  = -phi
    G0[7,7]  = -sig/(1-h)
    G0[7,20] = 1+phi

    G1[7,7] = -(sig*h)/(1-h)

    # Equation 8  (phillips curve - importing)

    G0[8,6]  = -bet
    G0[8,13] = 1+bet*delta_I
    G0[8,11] = -(1-theta_I)*(1-theta_I*bet)/theta_I
    G0[8,22] = -1

    G1[8,13] = delta_I

    Pi[8,5]  = -bet

    # Equation 9 (relationship pi and tot)

    G0[9,12] = -1
    G0[9,10] = -alfa

    G1[9,10] = -alfa
    G1[9,2]  = -1

    # Equation 10 (bugdet constraint)

    G0[10,21] = 1
    G0[10,10] = alfa
    G0[10,11] = alfa
    G0[10,9]  = -1
    G0[10,7]  = 1

    G1[10,21] = 1/bet

    # Equation 11 (UIP)

    G0[11,4]  = -1
    G0[11,15] = 1
    G0[11,8]  = 1
    G0[11,19] = -1
    G0[11,21] = chi
    G0[11,23] = -1

    Pi[11,6] = -1

    # Equation 12 (Taylor RMI)

    G0[12,8]  = 1
    G0[12,16] = -rho_1
    G0[12,9]  = -(1-rho_1-rho_2)*lambda_y
    G0[12,15] = -(1-rho_1-rho_2)*lambda_s

    G1[12,16] = rho_2
    G1[12,2]  = (1-rho_1-rho_2)*lambda_pi
    G1[12,15] = -(1-rho_1-rho_2)*lambda_s

    Psi[12,5] = sigma_i

    ###################
    ## Foreign Block ##
    ###################

    # System (Equations 13, 14 and 15)

    G0[13,17] = 1
    G0[14,17] = a_0piy
    G0[14,18] = 1
    G0[15,17] = a_0iy
    G0[15,18] = a_0ipi
    G0[15,19] = 1

    G1[13,17] = a_1yy
    G1[13,18] = a_1ypi
    G1[13,19] = a_1yi
    G1[14,17] = a_1piy
    G1[14,18] = a_1pipi
    G1[14,19] = a_1pii
    G1[15,17] = a_1iy
    G1[15,18] = a_1ipi
    G1[15,19] = a_1ii

    Psi[13,6] = sigma_y_star
    Psi[14,7] = sigma_pi_star
    Psi[15,8] = sigma_i_star

    #################
    ## Shocks block #
    #################

    # Equation 16 (a)

    G0[16,20] = 1

    G1[16,20] = rho_a

    Psi[16,1] = sigma_a

    # Equation 17 (gamma)

    G0[17,3] = 1

    G1[17,3] = rho_gamma

    Psi[17,2] = sigma_gamma

    # Equation 18 (epsilon_cp)

    G0[18,22] = 1

    G1[18,22] = rho_cp

    Psi[18,3] = sigma_cp

    # Equation 19 (phi)

    G0[19,23] = 1

    G1[19,23] = rho_phi

    Psi[19,4] = sigma_phi


    ####################
    ## Identity block ##
    ####################

    # Equation 20 (c)

    G0[20,7] = 1
    G1[20,1] = 1

    # Equation 21 (s)

    G0[21,15] = 1
    G1[21,4]  = 1

    # Equation 22 (pi_D)

    G0[22,12] = 1
    G1[22,5]  = 1

    # Equation 23 (pi_I)

    G0[23,13] = 1
    G1[23,6]  = 1

    # Equation 24 (i)

    G0[24,16] = 1
    G1[24,8]  = 1

    nobs = 100
    nobs = size(data,1) # number of observations in data

    pp = size(G1,1) # number of endogenous vars

    pobs = 8
    pobs = size(data,2)

    nshocks = size(Psi,2) # number of shocks

    nexpec = size(Pi,2) # number of expectation errors

    sol = gensys(G0,G1,Psi,Pi; verbose = false)
    if sum(sol.eu) != 2
        return -99999999999
    end

    #Sig = zeros(p,p)
    #Sig[4,4] = par[6]

    G = zeros(pobs,pp)
    G[1,9]  = 1 # y
    G[2,2]  = 1 # pi(t+1)
    G[3,8]  = 1 # i
    G[4,10] = 1 # tot
    G[5,15] = 1 # s
    G[6,17] = 1 # y_star
    G[7,18] = 1 # pi_star
    G[8,19] = 1 # i_star

    A = sol.Theta1
    R = Diagonal(repeat([1e-8],pobs)::AbstractVector)
    #Sigma = zeros(p,p)
    #Sigma[20,20] = par[31]^2 # sigma_a^2
    #Sigma[3,3]   = par[32]^2 # sigma_gamma^2
    #$Sigma[22,22] = par[33]^2 # sigma_cp^2
    # Sigma[23,23] = par[34]^2 # sigma_phi^2
    # Sigma[8,8]   = par[40]^2 # sigma_i^2
    # Sigma[17,17] = par[16]^2 # sigma_y_star^2
    # Sigma[18,18] = par[17]^2 # sigma_pi_star^2
    # Sigma[19,19] = par[18]^2 # sigma_i_star^2
    #Q = sol.Theta2*Sigma*sol.Theta2'
    Q = sol.Theta2*sol.Theta2'

    kalman_res = Kalman(A,G,Q,R) #create a Kalman filter instance

    y_mean = mean(data,dims=1)
    y_var = var(data,dims=1)
    y_var = reshape(y_var,size(y_var,2)) # transposição de forma a evitar problemas
    #y_var = repeat([1],p)

    x_hat = repeat(y_mean,p) # initial mean of the state
    x_var = diagm(repeat(y_var,p)) # variance initial of state
    # x_var = x_var*x_var'
    set_state!(kalman_res,x_hat,x_var)

    fit = zeros(nobs,pobs) # antes de pobs (8 agora) estava 4

    llh = zeros(nobs)

    burnin_kalman = 10

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
    llh = llh[burnin_kalman+1:length(llh)]
    return sum(llh)
end
