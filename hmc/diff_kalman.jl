function grad_y(x_hat_l,grad_x_hat_l,G, dG)
    m = size(G,1)
    return transpose(kron(x_hat_l', I(m))*dG'-G*grad_x_hat_l')
end

function grad_x_hat(x_hat_l, dA,A,grad_x_hat_l, y, grad_y, K,dK)
    n = size(A,1)
    return transpose(kron(x_hat_l',I(n))*dA' + A*grad_x_hat_l' + kron(y',I(n))*dK' + K*grad_y')
end

function diff_K(A,P,G,S_l,dA,dG,dP,dS_l)
    m = size(G,1)
    n = size(A,1)
    P_inv = inv(P)
    Kmn = commutation_matrix(m,n)
    return transpose(kron(P_inv*G*S_l,I(n))*dA' + kron(P_inv*G,A)*dS_l'+kron(P_inv,A*S_l)*Kmn*dG' - kron(P_inv,A*S_l*G'*P_inv)*dP')
end

function diff_S(A,P,K,S_l,dA,dP,dK,dS_l,dQ)
    m = size(P,1)
    n = size(A,1)
    Knn = commutation_matrix(n,n)
    Kmn = commutation_matrix(m,n)
    b1 = (kron(A*S_l,I(n)) + kron(I(n),A*S_l)*Knn)*dA'
    b2 = (kron(K*P,I(n)) + kron(I(n),K*P)*Kmn)*dK'
    return transpose(b1 + kron(A,A)*dS_l' -b2 - kron(K,K)*dP' +duplication_matrix(n)*dQ')
end

function diff_P(G,S_l,dG,dS_l,dR)
    m = size(G,1)
    n = size(G,2)
    Kmn = commutation_matrix(m,n)
    b1 = (kron(G*S_l,I(m)) + kron(I(m),G*S_l)*Kmn)*dG'
    return transpose(b1 + kron(G,G)*dS_l' + dR')
end

function diff_ll(P,y,grad_y,dP)
    P_inv = inv(P)
    return -1/2*det(P)*vec(P_inv)'*dP' - y'*P_inv*grad_y' + 1/2*y'*P_inv*dP'*P_inv*y
end

function diff_S0(A,S0,dA,dQ)
    n = size(A,1)
    Knn = commutation_matrix(n,n)
    b1 = (kron(A*S0,I(n)) + kron(I(n),A*S0)*Knn)*dA'
    ba = I(n^2) - kron(A,A)
    bb = b1 +duplication_matrix(n)*dQ'
    return transpose(inv(ba)*bb)
end
