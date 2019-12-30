#### Hansen`s model and modified model
### By Gilberto Boaretto and Daniel Coutinho
##### License: MIT

## To run this file, make sure you have packages Plots and LaTeXStrings in the system
## The Plots are better viewed with some IDE like Jupyter or Atom.
## Otherwise, go to the repl, press ] and type add <name-of-the-package>
## Make sure that you have the gensys file

include(string(pwd(),"/src/gensys.jl")) #Change this to where you saved gensys

## A word of warning about our gensys implementation: we return the matrices from the model with a unique solution in the case of multiple solutions. This is done because the procedure is not a 100 percent debbuged and in some cases with unique solution we return a warning of no unique solution (set gamma = 0, for example)

## Parameters for the original Hansen model, following McCandless book.

theta = 0.36
delta = 0.025
beta = 0.99
A = 1.72
h0 = 0.58
gamma = 0.95
sig = 0.0712

#Hansen's model with indivisible hours

B = A*log(1-h0)/h0

r_bar = 1/beta-(1-delta) #This is the interest rate in the steady state. 

## The matrices Γ_0, Γ_1, Π and Ψ for Gensys:

G0 = zeros(7,7)
G1 = zeros(7,7)
Pi = zeros(7,2)
Psi = zeros(7)

G0[1,1] = 1
G0[1,2] = -r_bar*beta

G0[3,5] = 1

G0[7,7] = 1

###G1####

G1[1,1] = 1

G1[2,1] = 1
G1[2,3] = 1
G1[2,4] = -1

G1[3,1] = -(r_bar/theta-delta)
G1[3,4] = r_bar/theta
G1[3,5] = (1-delta)

G1[4,3] = 1-theta
G1[4,4] = -1
G1[4,5] = theta
G1[4,7] = 1

G1[5,2] = -1
G1[5,4] = 1
G1[5,5] = -1

G1[6,3] = -1
G1[6,4] = 1
G1[6,6] = -1

G1[7,7] = gamma

Pi[1,1] = 1
Pi[1,2] = -beta*r_bar

Psi[7] = 1

sol1 = gensys(G0,G1,Psi,Pi)

irf1 = irf(sol1,100,0.01) #yes, this takes the whole model and knows where to look in the object of the type of the model which matrices to use. Neat!

using Plots
using LaTeXStrings

plot(irf1[:,1], w = 2, label = "C")
plot!(irf1[:,2], w = 2, label = "r")
plot!(irf1[:,3], w = 2, label = "h")
plot!(irf1[:,4], w = 2, label = "y")
plot!(irf1[:,5], w = 2, label = "k")
plot!(irf1[:,6], w = 2, label = "w", line = :dash, color = "red")
hline!([0], color = "black", w = 2, label = "0") #zero is kind of hard to see

###########################
##### Modified model######
#########################

theta = 0.36
delta = 0.025
beta = 0.99
A = 1.72
h0 = 0.58
gamma = 0.95
sig = 1#3.18 #CRRA coeficient
xi = 0 # 0.832 #habit formation

r_bar = 1/beta-(1-delta)
B = A*log(1-h0)/h0

rho = sig/((1-xi)*(1-beta*xi))
kappa = beta*(1-beta*xi)*(1-xi)/(sig*(1+beta*xi^2))*r_bar

G0 = zeros(9,9)
G1 = zeros(9,9)
Pi = zeros(9,3)
Psi = zeros(9)

###### Filling the matrices Γ_0, Γ_1, Π and Ψ for Gensys ############

########## Eq 1: Euler ##############

G0[1,1] = -beta*xi/(1+beta*xi^2)
G0[1,2] = 1+beta*xi/(1+beta*xi^2)
G0[1,4] = -kappa

G1[1,3] = -xi/(1+beta*xi^2)
G1[1,2] = 1+xi/(1+beta*xi^2)

Pi[1,1] = -beta*xi/(1+beta*xi^2)
Pi[1,2] = 1+beta*xi/(1+beta*xi^2)
Pi[1,3] = -kappa

####Eq 2 and 3: dummy equations#############

G0[2,2] = 1
G1[2,1] = 1

G0[3,3] = 1
G1[3,2] = 1

########## Eq 4: Labour supply ###############

G1[4,1] = -rho*beta*xi
G1[4,2] = rho*(1+beta*xi^2)
G1[4,3] = -rho*xi
G1[4,8] = -1

Pi[4,2] = rho*beta*xi

########  Eq 5: Production Function ###########

G1[5,5] = (1-theta)
G1[5,6] = -1
G1[5,7] = theta
G1[5,9] = 1

########## Eq 6: capital flux ####################

G0[6,7] = 1

G1[6,2] = -(r_bar/theta - delta)
G1[6,6] = r_bar/theta
G1[6,7] = (1-delta)

############### Eq 7: capital return ####################

G1[7,4] = -1
G1[7,6] = 1
G1[7,7] = -1

############## Eq 8: wage #########################

G1[8,5] = -1
G1[8,6] = 1
G1[8,8] = -1

############## Eq 9: autoregressive shock ################

G0[9,9] = 1

G1[9,9] = gamma

Psi[9] = 1

sol2 = gensys(G0,G1,Psi,Pi)

G0_aux = G0
G1_aux = G1
Psi_aux = Psi
Pi_aux = Pi

irf2 = irf(sol2,100,0.01)

plot(irf2[:,1], w = 2, label = "C")
plot!(irf2[:,2], w = 2, label = "r")
plot!(irf2[:,5], w = 2, label = "h")
plot!(irf2[:,6], w = 2, label = "y")
plot!(irf2[:,7], w = 2, label = "k")
plot!(irf2[:,8], w = 2, label = "w", line = :dash, color = "red")
hline!([0], color = "black", w = 2, label = "0")


# Sanity check: with sig = 1 and xi = 0, the model is exactly the same as before

plot(irf2[:,1], w = 2, label = "Modified Model")
plot!(irf1[:,1], w = 2, label = "Hansen's Model")
title!("c")

plot(irf2[:,4], w = 2, label = "Modified Model")
plot!(irf1[:,2], w = 2, label = "Hansen's Model")
title!("r")

plot(irf2[:,5], w = 2, label = "Modified Model")
plot!(irf1[:,3], w = 2, label = "Hansen's Model")
title!("h")

plot(irf2[:,6], w = 2, label = "Modified Model")
plot!(irf1[:,4], w = 2, label = "Hansen's Model")
title!("y")

plot(irf2[:,7], w = 2, label = "Modified Model")
plot!(irf1[:,5], w = 2, label = "Hansen's Model")
title!("k")

plot(irf2[:,8], w = 2, label = "Modified Model")
plot!(irf1[:,6], w = 2, label = "Hansen's Model")
title!("w")

plot(irf2[:,9], w = 2, label = "Modified Model")
plot!(irf1[:,7], w = 2, label = "Hansen's Model")
title!(L"\lambda")
