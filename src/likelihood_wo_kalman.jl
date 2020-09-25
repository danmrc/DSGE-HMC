using Statistics
using LinearAlgebra
using Distributions

include(string(pwd(),"/src/gensys.jl"))
include(string(pwd(),"/hmc/diffs_estrut.jl"))
include(string(pwd(),"/misc/matrix_no_kalman.jl"))

# See Scjmitt-Grohé paper on evaluating likelihoods without the Kalman filter to get the notation

function dSx(h,Sx,dh,dQ)
    m = size(h,1)
    Kmm = commutation_matrix(m,m)
    D = duplication_matrix(m)

    b1 = (I(m^2) - kron(h,h))
    b2 = kron(h*Sx,I(m)) + kron(I(m),h*Sx)*Kmm
    res = -inv(b1)*(b2*dh' - D*dQ')

    return res
end

function kron_and_sum(M,k)
    res = zeros(size(M).^2)
    for j in 1:k
        term = kron(M^(k-j),(M')^j)
        res += term
    end
    return res
end

function dSy(g,h,Sx,dSx,dh,j) #j is the power of h
    m = size(h,1)
    Kmm = commutation_matrix(m,m)
    h_pow = h^j
    ks = kron_and_sum(h,j)
    b1 = kron(g,g*h*Sx)*ks
    b2 = kron(g*h_pow*Sx,g)
    b3 = kron(g*h_pow,g*h)
    res = (b1*Kmm+b2)*dh' + b3*dSx
    return res
end

function diff_ll(P,y,dP)
    P_inv = inv(P)
    return -1/2*vec(P_inv)'*dP + 1/2*kron(y'*P_inv,y'*P_inv)*dP
end

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
    l = size(data,2)

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
        return -Inf, repeat([0],length(par))
    end

    #Sig = zeros(p,p)
    #Sig[4,4] = par[6]

    G = zeros(1,p)
    G[1,2] = 1

    A = sol.Theta1
    R = [0] .+ 1e-8
    Q = par[6]^2*sol.Theta2*sol.Theta2'

    d_reduc = diff_mod(par,l)
    d_reduc = d_reduc'
    dA = d_reduc[:,1:16]
    dA[dA .< 10*eps()] .= 0 #force whatever is bellow the eps to become zero
    dG = d_reduc[:,17:20]
    dQ = d_reduc[:,21:30]
    dQ[dQ .< 10*eps()] .= 0
    dR = zeros(size(par,1),1)

    Sx = solve_lyapunov_vec(A,Q)

    S = build_variance(G,A,Q,nobs)
    S = copy(S)

    dSx_mat = dSx(A,Q,dA,dQ)
    dSy_foo(j) = dSy(G,A,Sx,dSx_mat,dA,j)

    d_vecS = map(dSy_foo,1:(nobs))

    sel_mat = Toeplitz(1:nobs,1:nobs)
    sel_mat = Int.(sel_mat)

    dS_mat = d_vecS[sel_mat]
    dS_mat = vec(dS_mat)
    dS_mat = mapreduce(x->dS_mat[x],vcat,1:nobs^2)

    dll = diff_ll(S,data,dS_mat)

    dist = MvNormal(S)

    llh = logpdf(dist,data)

    return llh,dll
end
