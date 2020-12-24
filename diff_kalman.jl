include("aux_matrix.jl")

mutable struct diff_mats
        dA
        dS_l
        dR
        dx_hat_l
        dQ
end

function grad_y(x_hat_l,grad_x_hat_l,G)
    m = size(G,1)
    return -G*grad_x_hat_l'
end

function grad_x_hat(x_hat_l,A,K, err,grad_x_hat_l, dA, dK,d_err)
    n = size(A,1)
    return kron(x_hat_l',I(n))*dA + A*grad_x_hat_l' + kron(err',I(n))*dK + K*d_err
end

function diff_K(A,P,G,S_l,dA,dP,dS_l)
    m = size(G,1)
    n = size(A,1)
    P_inv = inv(P)
    aa = kron(P_inv*G*S_l,I(n))*dA
    bb = kron(P_inv*G,A)*dS_l
    cc = kron(P_inv,A*S_l*G'*P_inv)*dP
    return  aa +  bb - cc
end

function diff_S(A,P,K,S_l,dA,dP,dK,dS_l,dQ)
    m = size(P,1)
    n = size(A,1)
    Knn = commutation_matrix(n,n)
    Kmn = commutation_matrix(m,n)
    b1 = (kron(A*S_l,I(n)) + kron(I(n),S_l*A')*Knn)*dA
    b2 = (kron(K*P,I(n)) + kron(I(n),K*P)*Kmn)*dK
    return b1 + kron(A,A)*dS_l -b2 - kron(K,K)*dP +dQ
end

function diff_P(G,S_l,dS_l,dR)
    return kron(G,G)*dS_l + dR'
end

function diff_ll(P,eta,grad_eta,dP)
    P_inv = inv(P)
    if isempty(size(P_inv))
        return -1/2*P_inv*dP - eta'*P_inv*grad_eta + 1/2*kron(eta'*P_inv,eta'*P_inv)*dP
    else
        return -1/2*vec(P_inv)*dP - eta'*P_inv*grad_eta + 1/2*kron(eta'*P_inv,eta'*P_inv)*dP
    end
end

function diff_S0(A,S0,dA,dQ)
    n = size(A,1)
    Knn = commutation_matrix(n,n)
    b1 = (kron(A*S0,I(n)) + kron(I(n),A*S0)*Knn)*dA
    if n == 1
        ba = 1 - kron(A,A)
    else
        ba = I(n^2) - kron(A,A)
    end
    bb = b1 + dQ
    return inv(ba)*bb
end

function diff_kalman(kalman_inst,diff_mats,data)
    med_state = kalman_inst.cur_x_hat
    var_state = kalman_inst.cur_sigma

    n = size(kalman_inst.A,1)

    P = kalman_inst.G*var_state*kalman_inst.G' + kalman_inst.R

    gain = (kalman_inst.A*var_state*kalman_inst.G')*inv(P)

    d_p = diff_P(kalman_inst.G,var_state,diff_mats.dS_l,diff_mats.dR)

    d_gain = diff_K(kalman_inst.A,P,kalman_inst.G,var_state,diff_mats.dA,d_p,diff_mats.dS_l)

    g_eta = grad_y(med_state,diff_mats.dx_hat_l,kalman_inst.G)

    obs_err = data - kalman_inst.G*med_state

    grad_err = -kalman_inst.G*diff_mats.dx_hat_l'

    dll = diff_ll(P,obs_err,grad_err,d_p)

    llh = -1/2*(n*log(2*pi) + logdet(P) + obs_err'*inv(P)*obs_err)

    g_med_state = grad_x_hat(med_state,kalman_inst.A,gain, obs_err,diff_mats.dx_hat_l, diff_mats.dA, d_gain,g_eta)

    dS = diff_S(kalman_inst.A,P,gain,var_state,diff_mats.dA,d_p,d_gain,diff_mats.dS_l,diff_mats.dQ)

    return llh,dll,dS,g_med_state
end
