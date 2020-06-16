#####################################
#### Lots of things to change
# dvecA and and dvecΩ should become their closed values expressions
#Implement the commutation matrix
# Change dvec(Gamma_1) etc to expression that can be evaluated
#change every reshape for vecc, defined in the aux file

using ForwardDiff

include(string(pwd(),"/src/gensys.jl"))
include(string(pwd(),"/misc/aux_matrix.jl"))

function dF1_tau(Gamma_0, Gamma_1, A,l)
    # l is the number of observables
    # n is the number of shocks
    m = size(Gamma_0,1)
    nn = Int(m*(m+1)/2)
    dvecA = [I(m^2) zeros(m^2,m*l+nn)]
    (kron(-A', Gamma_1) - kron(I(m), Gamma_0) + kron(I(m),Gamma_1*A))* dvecA
end

function dF1_theta(dGamma_0,dGamma_1,dGamma_2,A)
    m = size(A,1)
    #dv_size = size(dGamma_0,1)*size(dGamma_0,2)
    #dGamma_0 = vecc(dGamma_0)
    #dGamma_1 = vecc(dGamma_1,dv_size,1)
    #dGamma_2 = vecc(dGamma_2)
    kron(A',I(m))* dGamma_0 - kron(A'*A',I(m))* dGamma_1 - dGamma_2
end

#this is a vech
function dF2_tau(Gamma_0, Gamma_1, Gamma_2, A, Omega,l)
    m = size(Gamma_0,1)
    b_aux = (Gamma_0 - Gamma_1*A)*Omega
    Dm = duplication_matrix(m)
    Kmm = commutation_matrix(m,m)
    nn = Int(m*(m+1)/2)

    dvecA = [I(m^2) zeros(m^2,m*l+nn)]
    dvecOmega = [zeros(nn,m^2+m*l) I(nn)]
    res = (kron(b_aux,Gamma_1) + kron(Gamma_1, b_aux)*Kmm)*dvecA + kron(Gamma_0 - Gamma_1*A,Gamma_0 - Gamma_1*A)*Dm*dvecOmega
    return(pinv(Dm)*res)
end

function dF2_theta(Gamma_0, Gamma_1, Gamma_3, dGamma_0, dGamma_1, dGamma_3, Omega, A)
    m = size(Gamma_0,1)
    b_aux = Gamma_0 - Gamma_1*A
    Dm = duplication_matrix(m)
    Kmm = commutation_matrix(m,m)

    dv_size = size(dGamma_0,1)*size(dGamma_0,2)
    #dGamma_0 = vecc(dGamma_0)
    #dGamma_1 = vecc(dGamma_1)
    #dGamma_2 = vecc(dGamma_2)

    b1 = kron(b_aux*Omega, I(m)) + kron(I(m), b_aux*Omega)*Kmm
    b2 = kron(b_aux*Omega*A',I(m)) + kron(I(m),b_aux*Omega)*Kmm
    b3 = kron(Gamma_3,I(m)) + kron(I(m),Gamma_3)*Kmm
    res = b1*dGamma_0 + b2*dGamma_1 + b3*dGamma_3
    res = pinv(Dm)*res
end

function dtheta(Gamma_0,Gamma_1,Gamma_2,Gamma_3, dGamma_0, dGamma_1, dGamma_2, dGamma_3, A, Omega,l)
    Df_theta = [dF1_theta(dGamma_0,dGamma_1,dGamma_2,A);dF2_theta(Gamma_0, Gamma_1, Gamma_3, dGamma_0, dGamma_1, dGamma_3, Omega, A)]
    Df_tau = [dF1_tau(Gamma_0, Gamma_1, A,l); dF2_tau(Gamma_0, Gamma_1, Gamma_2, A, Omega,l)]
    return pinv(Df_theta)*Df_tau
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

PSI_foo(par) = [0; 0; 0; 1]

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
    Omega = gen_sol.Theta2*gen_sol.Theta2'
    diff = dtheta(Gamma0,Gamma1,Gamma2,Gamma3, dGamma0, dGamma1, dGamma2, dGamma3, A, Omega,l)
    return diff
end
