using Plots

include(string(pwd(), "/src/interpreter.jl"))
include(string(pwd(), "/src/gensys.jl"))
include(string(pwd(), "/galis_example.jl"))

@polyvar pi pi_e y y_e i v v_l ep

#variables = [pi pi_e y y_e v v_l ep]

#Equations of the model

eq1 = pi - bet*pi_e - kappa*y
eq2 = y + 1/sig*(i - pi_e) - y_e
eq3 = i - phi_pi*pi - phi_y*y - v_l
eq4 = v - rho_v*v_l - ep

#Dictionary with all the equations

model = Dict("IS"=>eq2,"Phillips"=>eq1,"Taylor"=>eq3,"Ar Shock"=>eq4)

modelgensys = model2gensys(model,[ep])

sol = gensys(modelgensys)

fris = irf(sol,15,0.25)

plot(4*fris[:,1],w = 3)
plot!(4*irfs_true[2:15,1], line = :dash,w = 3)

plot(fris[:,2],w = 3)
plot!(irfs_true[2:15,2], line = :dash,w = 3)

plot(4*fris[:,3],w = 3)
plot!(4*irfs_true[2:15,4], line = :dash,w = 3)

plot(fris[:,3],w = 3)
plot!(irfs_true[2:15,4], line = :dash,w = 3)
