using Plots

include(string(pwd(), "/src/gensys_v2.jl"))
include(string(pwd(), "/gali_bayesian.jl"))

sol = gensys(GAMMA_0,GAMMA_1,PSI,PI)

irfs = irf(sol,15,0.25)

Plots.plot(4*irfs[:,1], label = "Inflação",w=3)
Plots.plot!(4*irfs_true[2:15,1], label = "Inflation True", line = :dash,w=3)

Plots.plot(irfs[:,2], label = "Output Gap",w=3)
Plots.plot!(irfs_true[2:15,2], label = "Output Gap True",line = :dash,w=3)

Plots.plot(4*irfs[:,3], label = "Juros",w=3)
Plots.plot!(4*irfs_true[2:15,3], label = "Juros True",w=3,line = :dash)

Plots.plot(irfs[:,4], label = "v",w=3)
Plots.plot!(irfs_true[2:15,4], label = "v True",w=3,line = :dash)
