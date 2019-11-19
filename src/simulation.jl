include(string(pwd(),"/src/gensys.jl"))

function simulate_dsge(G0,G1,Psi,Pi,Sigma = I,n; burn_in = 500)
    #G0,G1,Psi,Pi are the usual suspects from gensys
    #Sigma is the variance-covariance matrix of errors
    #Currently: Choleski Sigma and multiply randn output by it, I have no idea if this is the right thing to do
    #n is the sample size
    n_shocks = size(Pi,2) #number of columns of Pi = number of inovations in the system
    total = n + burn_in

    if Sigma == I
        Sigma = Array(I,n_shocks,n_shocks)
    end

    sol = gensys(G0,G1,Psi,Pi)

    if sum(sol.eu !=2)
        @error "Non existent or not unique solution"
    end

    C = cholesky(Sigma)
    C = C.U' #we want the transpose to get things right
    shocks = C*randn(n_shocks,total)
    shocks = shocks' #columns are different shocks, lines different dates, change my mind

    resul = zeros(total,size(G0,1))

    for j in 2:total
        resul[j,:] = sol.Theta1*resul[j-1,:] + sol.Theta2*shocks[j,:]
    end

    resul = resul[burn_in+1:total,:]
    return resul
end
