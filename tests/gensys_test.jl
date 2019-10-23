include(string(pwd(), "/src/gensys.jl"))
include(string(pwd(), "/gali_bayesian.jl"))

T1,T2,eu = gensys(GAMMA_0,GAMMA_1,PSI,PI)

irfs = irf(T1,T2,15,0.25)

using Plots

plot(4*irfs[:,1], label = "Inflação")
plot!(4*irfs_true[2:15,1], label = "Inflation True")
plot(irfs[:,2], label = "Output Gap")
plot(irfs[:,3], label = "Juros")
plot(irfs[:,4], label = "v")
