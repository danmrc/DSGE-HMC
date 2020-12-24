using ForwardDiff,SparseArrays

include(string(pwd(),"/gensys.jl"))
include(string(pwd(),"/aux_matrix.jl"))

function dF1_theta(A,dGamma_0,dGamma_1,dGamma_2)
    m = size(A,1)
    return kron(A'*I,I(m))*dGamma_0 - kron(A'^2,I(m))*dGamma_1 - dGamma_2
end

function dF1_A(A,Gamma_0,Gamma_1)
    m = size(A,1)
    b1 = Gamma_0 - Gamma_1*A
    return kron(I(m),b1) - kron(A'*I,Gamma_1)
end

function dA_theta(A,Gamma_0,Gamma_1,dGamma_0,dGamma_1,dGamma_2)
    dvecA = dF1_A(A,Gamma_0,Gamma_1)
    dTheta = dF1_theta(A,dGamma_0,dGamma_1,dGamma_2)
    return -dvecA\Array(dTheta)
end

function dB_theta(Gamma_0,Gamma_1,Gamma_3,A,dGamma_0,dGamma_1,dGamma_2,dGamma_3)
    dA = dA_theta(A,Gamma_0,Gamma_1,dGamma_0,dGamma_1,dGamma_2)
    dA[abs.(dA) .< 10*eps()] .= 0
    dA = sparse(dA)

    m = size(A,1)
    n = size(Gamma_3,2)

    b0 = Array(Gamma_0 - Gamma_1*A)
    b0[abs.(b0) .< 10*eps()] .= 0
    b0 = pinv(b0)

    b1 = kron((b0*Gamma_3)', b0)
    b2 = dGamma_0 - kron(A',I(m))*dGamma_1 - kron(I(m),Gamma_1)*dA

    return -b1*b2 + kron(I(n),b0)*dGamma_3
end

## These are the gensys matrices

#order to par
#1 alfa
#2 beta
#3 epsilon
#4 theta
#5 sig
#6 sigma: this is the std dev of the innovation
#7 phi
#8 phi_pi
#9 phi_y
#10 rho_v


GAMMA_0_foo(par) = [par[2]  0    0  0;
           1    par[5]  0  0;
           0    0    0  0;
           0    0    0  1]

function GAMMA_1_foo(par)
    THETA = (1-par[1])/(1-par[1]+par[1]*par[3])
    lamb = (1-par[4])*(1-par[2]*par[4])/par[4]*THETA
    kappa = lamb*(par[5]+(par[7]+par[1])/(1-par[1]))
    res = [ 1       -kappa  0   0;
            0        par[5]    1   0;
           -par[8]  -par[9]  1  -1;
            0        0      0   par[10]]
            return res
end

PSI_foo(par) = [0; 0; 0; par[6]]

PI_foo(par) = [par[2]  0;
      1    par[5];
      0    0;
      0    0]

## These are the matrices for the same model written as iskrev (2008)

#order to par
#1 alfa
#2 beta
#3 epsilon
#4 theta
#5 sig
#6 sigma: this is the std dev of the innovation
#7 phi
#8 phi_pi
#9 phi_y
#10 rho_v

function Gamma_0(par)
    THETA = (1-par[1])/(1-par[1]+par[1]*par[3])
    lamb = (1-par[4])*(1-par[2]*par[4])/par[4]*THETA
    kappa = lamb*(par[5]+(par[7]+par[1])/(1-par[1]))
    res = [1 -kappa 0 0;
            0       1   1/par[5] 0;
            -par[8] -par[9] 1  -1;
            0           0    0   1;
            ]
    return res
end

Gamma_1(par) = [par[2] 0 0 0;
                1/par[5] 1 0 0;
                0        0 0 0;
                0        0 0 0]

Gamma_2(par) =[0 0 0 0;
               0 0 0 0;
               0 0 0 0;
               0 0 0 par[10]]

Gamma_3(par) = [0;
                0;
                0;
               par[6]]

## Diff for the model
# Now it will be a #estrut par x ## reduce form par
# In which the reduced form depends of 3 matrices, in the following order
# A, that governs the auto regressive part of the reduce form: first 16 cols
# C that govens the observation matrix in the Kalman Filter (usually should be all zeros): 4 columns
# Omega, that is BB' in which B governs the shock transmission: remaining 10 columns (it is a vech!)

function diff_mod(par;spar=true)
    Gamma0 = Gamma_0(par)
    Gamma1 = Gamma_1(par)
    Gamma2 = Gamma_2(par)
    Gamma3 = Gamma_3(par)

    G0 = GAMMA_0_foo(par)
    G1 = GAMMA_1_foo(par)
    Psi = PSI_foo(par)
    Pi = PI_foo(par)

    dGamma0 = ForwardDiff.jacobian(Gamma_0,par)
    dGamma1 = ForwardDiff.jacobian(Gamma_1,par)
    dGamma2 = ForwardDiff.jacobian(Gamma_2,par)
    dGamma3 = ForwardDiff.jacobian(Gamma_3,par)

    gen_sol = gensys(G0,G1,Psi,Pi, verbose = false)
    A = gen_sol.Theta1

    if spar == true #use sparse arrays
        dGamma0 = sparse(dGamma0)
        dGamma1 = sparse(dGamma1)
        dGamma2 = sparse(dGamma2)
        dGamma3 = sparse(dGamma3)

        Gamma0 = sparse(Gamma0)
        Gamma1 = sparse(Gamma1)
        Gamma2 = sparse(Gamma2)
        Gamma3 = sparse(Gamma3)

        A[abs.(A) .< 10*eps()] .= 0
        A = sparse(A)
    end

    dA = dA_theta(A,Gamma0,Gamma1,dGamma0,dGamma1,dGamma2)
    dB = dB_theta(Gamma0,Gamma1,Gamma3,A,dGamma0,dGamma1,dGamma2,dGamma3)

    if spar == true
        dA = sparse(dA)
    end

    return dA,dB
end
