include(string(pwd(),"/github/DSGE-HMC/log_like.jl"))

data = randn(100,3)
A = [0.5 0.3 0.2;0.4 0.2 -0.1; -0.5 0.8 -0.9]

loglike(A,data)
