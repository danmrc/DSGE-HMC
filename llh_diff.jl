include("gensys.jl")
include("diffs_estrut_v3.jl")
include("diff_kalman.jl")

using QuantEcon

function solve_lyapunov_vec(A,Q)
    aux = (I - kron(A,A))
    vec_ans = \(aux,vec(Q))
    res = reshape(vec_ans,size(Q))
    return res
end

function log_like_dsge(par,data)

    nobs = size(data,1)
    l = size(data,2)
    n_pars = size(par,1)

    G0 = GAMMA_0_foo(par)
    G1= GAMMA_1_foo(par)
    PI = PI_foo(par)
    PSI = PSI_foo(par)

    p = size(G0,1) #number of endogenous variables

    mod = gensys(G0,G1,PSI,PI,verbose=false)

    if sum(mod.eu) != 2
        return -Inf, zeros(size(par))
    end

    A = mod.Theta1
    B = mod.Theta2

    A[abs.(A) .< 10*eps()] .= 0
    B[abs.(B) .< 10*eps()] .= 0

    Q = B*B'
    R = [1e-3]
    G = zeros(1,p)
    G[1,2] = 1

    dA,dB = diff_mod(par)
    dG = zeros(size(par,1),4)
    dQ = kron(B,I(p))*dB + kron(I(p),B)*commutation_matrix(p,1)*dB #vec, not vech
    dR = zeros(size(par,1),1)

    x0 = zeros(p)
    V0 = solve_lyapunov_vec(A,Q)

    llh = zeros(nobs)
    dll = zeros(nobs,n_pars)

    kalman_inst = Kalman(A,G,Q,R)

    set_state!(kalman_inst,x0,V0)
    d_S0 = diff_S0(A,V0,dA,dQ)*1e-1
    dx_hat_l = zeros(n_pars,p)

    diff_mat = diff_mats(dA,d_S0,dR,dx_hat_l,dQ)

    for j = 1:nobs
        llh[j],dll[j,:],dS,∇x_l = diff_kalman(kalman_inst,diff_mat,data[j,:])
        diff_mat.dS_l = dS
        diff_mat.dx_hat_l = ∇x_l'
        QuantEcon.update!(kalman_inst,data[j,:])
    end

    return sum(llh),sum(dll,dims=1)

end
