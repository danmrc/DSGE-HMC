## MAGIJ: Matrix Assistance to Gensys In Julia
## WARNING: Higlhy experimental
##

##TODO: Allow keys different of numbers in dict (or, how to index dict?)
## TODO: Automatic Creation of expectations shock and foward variables

using DynamicPolynomials
using Plots

#Parameters creation

bet = 0.99
sig = 1
phi = 1
alfa = 1/3
epsilon = 6
theta = 2/3
phi_pi = 1.5
phi_y = 0.5/4
rho_v = 0.5

THETA = (1-alfa)/(1-alfa+alfa*epsilon)
lamb = (1-theta)*(1-bet*theta)/theta*THETA
kappa = lamb*(sig+(phi+alfa)/(1-alfa))

#Variables

@polyvar pi pi_e y y_e i v v_l ep

#variables = [pi pi_e y y_e v v_l ep]

#Equations of the model

eq1 = pi - bet*pi_e - kappa*y
eq2 = y + 1/sig*(i - pi_e) - y_e
eq3 = i - phi_pi*pi - phi_y*y - v_l
eq4 = v - rho_v*v_l - ep

#Dictionary with all the equations

model = Dict(1=>eq1,2=>eq2,3=>eq3,4=>eq4)

model = model2gensys(model,[ep])

#What variables appear in which model

varis = Dict()

for i in keys(model)
    merge!(varis,Dict(i=>variables(model[i])))
end

#Defyning the auxilary variables

@polyvar nu_pi nu_y pi_f y_f

pi_e_aux = pi_f - nu_pi
y_e_aux = y_f - nu_y

foward_variables = [pi_f y_f v]
backward_variables = [pi y v_l i]
shocks = [ep]
expec_errors = [nu_pi nu_y]

f_string = [repr(i) for i in foward_variables]
expec_vars_test = match.(r"_f",f_string)
expec_vars_test = expec_vars_test .!= nothing
expec_vars = f_string[expec_vars_test]
stripped_vars = [rstrip(vv,['_','f']) for vv in expec_vars]
foward_vars = string.(stripped_vars,"_f")

bb = "@polyvar"

for aa in foward_vars
    global bb = join([bb,aa]," ")
end


for i in keys(model)
    if sum(pi_e .== varis[i]) == 1
        model[i] = subs(model[i],pi_e => pi_e_aux)
    end
    if sum(y_e .== varis[i]) == 1
        model[i] = subs(model[i],y_e => y_e_aux)
    end
end

vari_all = Vector()

for i in keys(varis)
    union!(vari_all,variables(model[i]))
end

#Building each matrix

Gamma1 = zeros(length(model),length(model))
Gamma0 = zeros(length(model),length(model))
Psi = zeros(length(model),length(shocks))
Pi =  zeros(length(model),length(expec_errors))

#Gamma 0

for j in 1:length(foward_variables)
    for i in keys(model)
        Gamma0[i,j] = -coefficient(model[i],foward_variables[j])
    end
end

#Gamma 1

for j in 1:length(backward_variables)
    for i in keys(model)
        Gamma1[i,j] = coefficient(model[i],backward_variables[j])
    end
end

#Psi

for j in 1:length(shocks)
    for i in keys(model)
        Psi[i,j] = coefficient(model[i],shocks[j])
    end
end

#Pi

for j in 1:length(expec_errors)
    for i in keys(model)
        Pi[i,j] = coefficient(model[i],expec_errors[j])
    end
end

sol = gensys(model.Gamma0,model.Gamma1,model.Psi,model.Pi)

fris = irf(sol,15,0.25)

plot(4*fris[:,1],w = 3)
plot!(4*irfs_true[2:15,1], line = :dash,w = 3)

plot(fris[:,2],w = 3)
plot!(irfs_true[2:15,2], line = :dash,w = 3)

plot(4*fris[:,4],w = 3)
plot!(4*irfs_true[2:15,3], line = :dash,w = 3)

plot(fris[:,3],w = 3)
plot!(irfs_true[2:15,4], line = :dash,w = 3)
