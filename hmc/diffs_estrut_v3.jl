using ForwardDiff

include(string(pwd(),"/src/gensys.jl"))
include(string(pwd(),"/misc/aux_matrix.jl"))

function dF1_theta(A,dGamma_0,dGamma_1,dGamma_2)
    m = size(A,1)
    return kron(A',I(m))*dGamma_0 - kron(A'^2,I(m))*dGamma_1 - dGamma_2
end

function dF1_A(A,Gamma_0,Gamma_1)
    m = size(A,1)
    b1 = Gamma_0 - Gamma_1*A
    return kron(I(m),b1) - kron(A',Gamma_1)
end

function dA_theta(A,Gamma_0,Gamma_1,dGamma_0,dGamma_1,dGamma_2)
    dvecA = dF1_a(A,dGamma_0,Gamma_1)
    dtheta = dF1_theta(A,dGamma_0,dGamma_1,dGamma_2)
    return -inv(dvecA)*dTheta
end

function dB_theta(Gamma_0,Gamma_1,Gamma_3,A,dGamma_0,dGamma_1,dGamma_2,dGamma_3)
    dA = dA_theta(A,Gamma_0,Gamma_1,dGamma_0,dGamma_1,dGamma_2)

    m = size(A,1)

    b0 = Gamma_0 - Gamma_1*A
    b0 = inv(b0)

    b1 = kron((b0*Gamma_3)', b0)
    b2 = dGamma_0 - kron(A',I(m))*dGamma_1 - kron(I(m),Gamma_1)*dA

    return -b1*b2 + kron(I(m),b0)*dGamma_3
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
    res = [par[1] -kappa 0 0;
            0       1   1/par[5] 0;
            -par[8] -par[9] -1  -1;
            0           0    0   1
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

Gamma_3(par) = [0 0 0 0;
               0 0 0 0;
               0 0 0 0;
               0 0 0 par[6]]

## Diff for the model
# Now it will be a #estrut par x ## reduce form par
# In which the reduced form depends of 3 matrices, in the following order
# A, that governs the auto regressive part of the reduce form: first 16 cols
# C that govens the observation matrix in the Kalman Filter (usually should be all zeros): 4 columns
# Omega, that is BB' in which B governs the shock transmission: remaining 10 columns (it is a vech!)

function diff_mod(par,l)
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
    dA = dA_theta(A,Gamma_0,Gamma_1,dGamma_0,dGamma_1,dGamma_2)
    dB = dB_theta(Gamma_0,Gamma_1,Gamma_3,A,dGamma_0,dGamma_1,dGamma_2,dGamma_3)

    return dA,dB
end
