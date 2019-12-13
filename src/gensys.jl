### Gensys Implementation
### Algorithm by Christopher A. Sims
### Implementation by Gilberto Boaretto and Daniel Coutinho
### License: MIT

using LinearAlgebra

struct Sims
    Theta1
    Theta2
    Theta3
    eu
end

function gensys(G0,G1,Psi,Pi;verbose = true, tol = 1e-12)
    decomp_1 =
    try
        schur(G0,G1)
    catch
        eu = (-1,-1)
        Theta1 = zeros(n,n)
        Theta2 = zeros(n,n)
        Theta3 = zeros(n,n)
        ans = Sims(Theta1,Theta2,Theta3,eu)
        return ans
        if verbose
            @warn "Unknown error. Probably LAPACK Exception 2. Skipping."
        end
    end
    gen_eigen = abs.(decomp_1.beta ./ decomp_1.alpha)
    ordschur!(decomp_1, gen_eigen .< 1)
    n = size(G0,1)
    ns = findfirst(sort(gen_eigen) .> 1) #finding the number of stable roots: find first unstable root
    if isnothing(ns)
        ns=1
    else
        ns = ns -1
    end
    nu = n - ns
    S11 = decomp_1.S[1:ns,1:ns]
    S12 = decomp_1.S[1:ns,(ns+1):n]
    S22 = decomp_1.S[(ns+1):n,(ns+1):n]

    T11 = decomp_1.T[1:ns,1:ns]
    T12 = decomp_1.T[1:ns,(ns+1):n]
    T22 = decomp_1.T[(ns+1):n,(ns+1):n]

    Qt = decomp_1.Q'

    Q1 = Qt[1:ns,:]
    Q2 = Qt[(ns+1):n,:]

    Q2Pi = Q2*Pi #This is equation 2.25 in p. 46 Miao (2014)

    m = size(Q2Pi,2)

    svd_Q2Pi = svd(Q2Pi)
    svd_aux = svd_Q2Pi.S #S is a vector
    r = sum(abs.(svd_aux) .> tol)

    #Checking existence and uniqueness see p. 46-47, Miao (2014)
    if m > r
        eu = [1;0]
        if verbose == true
            @warn "No Unique Solution"
        end
        Theta1 = zeros(n,n)
        Theta2 = zeros(n,n)
        Theta3 = zeros(n,n)
        ans = Sims(Theta1,Theta2,Theta3,eu)
        return ans
    elseif m < r
        eu = [0;0]
        Theta1 = zeros(n,n)
        Theta2 = zeros(n,n)
        Theta3 = zeros(n,n)
        ans = Sims(Theta1,Theta2,Theta3,eu)
        return ans
        if verbose == true
            @warn "No solution"
        end
    else
        eu = [1;1]
        if verbose == true
            @info "Unique and Stable Solution"
        end
        U1 = svd_Q2Pi.U[:,1:r]
        Xi = Q1*Pi*pinv(Q2Pi) #bottom of p 46 #change in 11 dec 2019: pinv instead of manually multiplying the elements

        Aux1 = S12-Xi*S22

        larg1 = size(Aux1,2)
        Aux2 = [S11 Aux1;zeros(larg1,size(S11,2)) Matrix(I,larg1,larg1)]
        larg2 = size(Aux2,2)
        larg2 = larg2 - size(T11,1)

        ##Matrices on top of p. 46

        Theta1 = [T11 T12-Xi*T22;zeros(larg2,n)]
        Theta1 = decomp_1.Z*inv(Aux2)*Theta1*decomp_1.Z'
        Theta2 = [Q1 - Xi*Q2;zeros(larg2,n)]
        Theta2 = decomp_1.Z*inv(Aux2)*Theta2*Psi
        Theta3 = zeros(n,n)
        ans = Sims(Theta1,Theta2,Theta3,eu)
        return ans#Theta1,Theta2,eu
    end
end

function irf(Theta1,Theta2,t,impulse)
    ans = zeros((t+1),size(Theta1,1))
    for j in 0:t
        ans[(j+1),:] = Theta1^j*Theta2*impulse
    end
    return ans
end

function irf(model::Sims,t,impulse)
    irf(model.Theta1,model.Theta2,t,impulse)
end
