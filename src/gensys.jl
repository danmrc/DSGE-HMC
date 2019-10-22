### Gensys Implementation
### Algorithm by Christopher A. Sims
### Implementation by Gilberto Boaretto and Daniel Coutinho
### License: MIT

using LinearAlgebra

function gensys(G0,G1,Psi,Pi)
    decomp_1 = schur(G0,G1)
    gen_eigen = abs.(decomp_1.alpha ./ decomp_1.beta)
    ordschur!(decomp_1, gen_eigen .< 1)
    n = size(G0,1)
    ns = findfirst(sort(gen_eigen) .> 1) -1 #finding the number of stable roots
    nu = n - ns
    S11 = decomp_1.S[1:ns,1:ns]
    S12 = decomp_1.S[1:ns,(ns+1):n]
    S22 = decomp_1.S[(ns+1):n,(ns+1):n]

    T11 = decomp_1.T[1:ns,1:ns]
    T12 = decomp_1.T[1:ns,(ns+1):n]
    T22 = decomp_1.T[(ns+1):n,(ns+1):n]

    Q1 = decomp_1.Q[1:ns,:]
    Q2 = decomp_1.Q[(ns+1):n,:]

    Q2Pi = Q2*Pi #This is equation2.25 in p. 46 Miao (2014)

    m = size(Q2Pi,2)

    svd_Q2Pi = svd(Q2Pi)
    r = size(svd_Q2Pi.S)[1] #S is a vector

    #Checking existence and uniqueness see p. 46-47, Miao (2014)
    if m > r
        eu = [1;0]
        @warn "Not Unique Solution"
    elseif m < r
        eu = [0;0]
        @warn "No solution"
    else
        @info "Unique and Stable Solution"
        U1 = svd_Q2Pi.U[:,1:r]
        Xi = Q1*Pi*svd_Q2Pi.V*inv(Diagonal(svd_Q2Pi.S))*U1'

    end



end
