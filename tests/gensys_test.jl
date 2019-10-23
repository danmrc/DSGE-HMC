using Plots

include(string(pwd(), "/src/gensys.jl"))
include(string(pwd(), "/gali_bayesian.jl"))

T1,T2,eu = gensys(GAMMA_0,GAMMA_1,PSI,PI)

irfs = irf(T1,T2,15,0.25)

plot(4*irfs[:,1], label = "Inflação")
plot!(4*irfs_true[2:15,1], label = "Inflation True")

plot(irfs[:,2], label = "Output Gap")
plot!(irfs_true[2:15,2], label = "Output Gap True")

plot(4*irfs[:,3], label = "Juros")
plot!(4*irfs_true[2:15,3], label = "Juros True")

plot(irfs[:,4], label = "v")
plot!(irfs_true[2:15,4], label = "v True")
