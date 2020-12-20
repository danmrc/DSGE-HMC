using LinearAlgebra

function diff_svd(A,dA) #for a single parameter
    A_svd = svd(A)
    U = A_svd.U
    V = A_svd.V
    S = A_svd.S

    dS = U'*dA*V
    s = size(dS,1)

    Up = triu(dS,1) #extract the upper triangular part of the dS without the diagonal
    Lo = tril(dS,-1) #extract the lower triangular part of the dS matrix without diagonal

    L2_bar = zeros(s,s)

    for i = 2:s,j=1:(i-1)
        L2_bar[i,j] = (Up[j,i]*S[j] + Lo[i,j]*S[i])/(S[i]^2- S[j]^2)
    end

    dV = V*(L2_bar' - L2_bar)
    return diag(dS),dV
end

function jacob_svd(A,dA)
    p = size(dA,2)
    if isempty(size(A))
        n = 1
        m = 1
    else
        n,m = size(A)
    end

    dV = zeros(n*m,p)
    dS = zeros(max(n,m),p)

    for k in 1:p
        dA_unvec = dA[:,k]
        dA_unvec = reshape(dA_unvec,n,m)
        dS_temp,dV_temp = diff_svd(A,dA_unvec)
        dS[:,k] = dS_temp
        dV[:,k] = vec(dV_temp)
    end

    return dS,dV

end

mutable struct svd_kalman_diff
    # ALL V MATRICES FROM THE SVD ARE STORE TRANSPOSED I.E. Vt
    A # state transition
    G #Observable transition

    #shock of observable variance
    S_Q
    V_Q

    #variance of the shock of the state
    S_R
    V_R

    state_mean

    #variance of the state

    S_var
    V_var

    dA
    dG

    dS_Q
    dV_Q

    dS_R
    dV_R

    dS_var
    dV_var

    diff_state

    D_re #this will always be the square root of what we need
    e_bar

end

function svd_kalman_diff(A,G,Q,R,dA,dG,dR,dQ)

    m,n = size(G)
    st_mean = zeros(n)
    st_var = zeros(n,n)

    dS_Q, dV_Q = jacob_svd(Q,dQ)
    dS_R, dV_R = jacob_svd(R,dR)

    diff_state = zeros(n^2,size(dA,2))

    svd_Q = svd(Q)
    svd_R = svd(R)

    S_R = diagm(sqrt.(svd_R.S))
    S_Q = diagm(sqrt.(svd_Q.S)[1:rank(Q)])

    V_R = svd_R.Vt
    V_Q = svd_Q.Vt

    dS_R = inv(kron(I(m),S_R) + kron(S_R,I(m)))*dS_R
    dS_Q = inv(kron(I(rank(Q)),S_Q) + kron(S_Q,I(rank(Q))))*dS_Q[1:rank(Q),:]

    D_re = zeros(m,m)
    e_bar = zeros(m)

    return svd_kalman_diff(A,G,S_Q,V_Q,S_R,V_R,st_mean,st_var,st_var,dA,dG,dS_Q,dV_Q,dS_R,dV_R,dS_Q,dV_Q,D_re,e_r)
end

function start!(svd_kalman_diff::svd_kalman_diff,x0,v0,dv0)
    m,n = size(svd_kalman_diff.G)

    svd_kalman_diff.state_mean = x0

    svd_v0 = svd(v0)
    S_var = diagm(sqrt.(svd_v0.S))
    svd_kalman_diff.S_var = S_var
    svd_kalman_diff.V_var = svd_v0.Vt

    svd_kalman_diff.diff_state = zeros(size(x0,1))

    dS_var,dV_var = jacob_svd(v0,dv0)

    dS_var = inv(kron(I(n),S_var) + kron(S_var,I(n)))*dS_var

    svd_kalman_diff.dS_var = dS_var
    svd_kalman_diff.dV_var = dV_var

    return nothing
end

function forecast!(svd_kalman_diff::svd_kalman_diff)
    m,n = size(svd_kalman_diff.G)
    Q = svd_kalman_diff.S_Q*svd_kalman_diff.V_Q
    A = svd_kalman_diff.A
    old_vari = svd_kalman_diff.S_var*svd_kalman_diff.V_var
    old_state = svd_kalman_diff.state_mean

    dSvar = svd_kalman_diff.diff_Svar
    dQvar = svd_kalman_diff.diff_Vvar

    new_state = A*old_state

    pre_var = [old_vari*A';Q]

    svd_var = svd(pre_var)

    svd_kalman_diff.state_mean = new_state
    svd_kalman_diff.S_var = diagm(svd_var.S)
    svd_kalman_diff.V_var = svd_var.Vt

    b1 = kron(A*svd_kalman_diff.V_var,I(n))*svd_kalman_diff.dS_var + kron(A,svd_kalman_diff.S_var)*svd_kalman_diff.dV_var + kron(I(n),old_vari)*commutation_matrix(n,n)*svd_kalman_diff.dA
    b2 = kron(svd_kalman_diff.V_Q,I(n))*svd_kalman_diff.dS_Q + kron(I,svd_kalman_diff.S_Q)*svd_kalman_diff.dV_Q
    dVar = [b1;b2]

    svd_kalman_diff.Svar,svd_kalman_diff.Vvar = jacob_svd(diagm(svd_var.S)*svd_var.Vt,dVar)

    svd_kalman_diff.diff_state = kron(new_state',I(n))*svd_kalman_diff.dA + A*svd_kalman_diff.diff_state

    return nothing
end

function update_state!(svd_kalman_diff::svd_kalman_diff,y)
    R = svd_kalman_diff.S_R*svd_kalman_diff.V_R
    old_var = svd_kalman_diff.S_var*svd_kalman_diff.V_var
    G = svd_kalman_diff.G
    old_state = svd_kalman_diff.state_mean
    d_var = kron(svd_kalman_diff.V_var',I(n))*svd_kalman_diff.dS_var + kron(I(n),svd_kalman_diff.S_var)*svd_kalman_diff.dV_var
    m,n = size(G)

    re_mat = [R;old_var*G]
    svd_re = svd(re_mat)

    K_bar = (old_var'*old_var)*G'*svd_re.Vt
    er = *(y - G*old_state)
    e_bar = svd_re.Vt*er

    svd_kalman_diff.e_bar = e_bar
    svd_kalman_diff.D_re = diagm(svd_re.S)

    new_state = old_state + K_bar*diagm(svd_re.S.^(-2))*e_bar

    K = K_bar*diagm(svd_re.S.^(-2))*svd_re.Vt

    pre_var = [old_var*(I - K*G)';R*K']

    svd_var = svd(pre_var)

    svd_kalman_diff.S_var = diagm(svd_var.S)
    svd_kalman_diff.V_var = svd_var.Vt

    b1 = kron(svd_kalman_diff.V_R,I(n))*svd_kalman_diff.dS_R + kron(I(n),svd_kalman_diff.S_R)*commutation_matrix(n,n)*svd_kalman_diff.dV_R

    b2 = kron(G*svd_kalman_diff.V_var,I(n))*svd_kalman_diff.dS_var + kron(G,svd_kalman_diff.S_var)*commutation_matrix(n,n)*svd_kalman_diff.dV_var + kron(I,svd_kalman_diff.S*svd_kalman_diff.V)*commutation_matrix(m,n)*svd_kalman_diff.dG

    diff_re = [b1;b2]
    diff_svd_re = jacob_svd(re_mat,diff_re)

    diff_k_bar = kron(svd_re.Vt*G,I(n))*d_var + kron(svd_re.Vt,old_var)*svd_kalman_diff.dG + kron(I(n),old_var*G')*diff_svd_re.dV

    diff_k = kron(svd_re.V*diagm(svd_re.S .^(-2)),I(n))*diff_k_bar - kron(svd_re.V,K_bar)*(kron(diagm(svd_re.S .^(-2)),diagm(svd_re.S)) + kron(diagm(svd_re.S),diagm(svd_re.S .^(-2))))*diff_svd_re.dS + kron(I(n),K_bar*diagm(svd_re.S .^(-2)))*diff_svd_re.dV

    e_diff = -(kron(old_state',I(m))*svd_kalman_diff.dG + G*svd_kalman_diff.diff_state)
    e_bar_diff = kron(er',I(m))*diff_svd_re.dV + svd_re.Vt*e_diff

    svd_kalman_diff.diff_state = svd_kalman_diff.diff_state + kron(e_bar'*diagm(svd_re.S .^(-2)),I(n))*diff_k_bar - kron(e_bar',K_bar)*(kron(diagm(svd_re.S .^(-2)),diagm(svd_re.S)) + kron(diagm(svd_re.S),diagm(svd_re.S .^(-2))))*diff_svd_re.dS + K_bar*diagm(svd_re.S .^(-2))*e_bar_diff

    b0 = I(n) - K*G
    b1 = kron(b0*svd_kalman_diff.V_var,I(n))*svd_kalman_diff.dS_var + kron(b0,svd_kalman_svd.S_var)*svd_kalman_diff.dV_var - kron(K,svd_kalman_diff.S_var*svd_kalman_diff.V_var)*commutation_matrix(m,n)*svd_kalman_diff.dG - kron(I(n),svd_kalman_diff.S_var*svd_kalman_diff.V_var*G')*commutation_matrix(n,m)*dK
    b2 = kron(K*svd_kalman_diff.V_R',I(m))*svd_kalman_diff.dS_R + kron(K,svd_kalman_diff.S_R)*svd_kalman_res.dQ_R + kron(I(n),svd_kalman_diff.D_R*svd_kalman_diff.Q_R)*commutation_matrix(n,m)*dK

    diff_var = [b1;b2]

    var_pre = [old_var*b0';svd_kalman_diff.S_R*svd_kalman_diff.V_R*K']

    svd_var = svd(var_pre)

    svd_kalman_diff.S_var = diagm(svd_var.S)
    svd_kalman_diff.V_var = svd_var.Vt

    svd_kalman_diff.dS_var,svd_kalman_diff.dV_var = svd_jacob(var_pre,diff_var)

    return nothing

end

function update_and_forecast!(svd_kalman_diff::svd_kalman_diff,y)
    update_state!(svd_kalman_diff,y)
    forecast!(svd_kalman_diff)
    return nothing
end
