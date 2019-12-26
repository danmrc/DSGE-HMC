include(string(pwd(),"/src/gensys.jl"))

theta = 0.36
delta = 0.025
beta = 0.99
A = 1.72
h0 = 0.58
gamma = 0.95
sig = 0.0712

B = A*log(1-h0)/h0

r_bar = 1/beta-(1-delta)

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

irf1 = irf(sol1,100,0.01)

using Plots

plot(irf1[:,1], w = 2, label = "C")
plot!(irf1[:,2], w = 2, label = "r")
plot!(irf1[:,3], w = 2, label = "h")
plot!(irf1[:,4], w = 2, label = "y")
plot!(irf1[:,5], w = 2, label = "k")
plot!(irf1[:,6], w = 2, label = "w", line = :dash, color = "red")
hline!([0], color = "black", w = 2)

####################

G0 = zeros(7,7)
G1 = zeros(7,7)
Pi = zeros(7,2)
Psi = zeros(7)

h_bar = - (1-theta)/(B*(1-delta*theta*beta/(1-beta*(1-delta))-1))
k_bar = (theta*beta/(1-beta*(1-delta)))^(1/(1-theta))*h_bar
y_bar = r_bar*k_bar/theta
c_bar = y_bar - delta*k_bar

G0[1,1] = 1
G0[1,2] = -r_bar*beta

G0[2,1] = 1
G0[2,3] = 1
G0[2,4] = -1

G0[3,5] = k_bar

G0[4,3] = -(1-theta)
G0[4,4] = 1
G0[4,5] = -theta
G0[4,7] = -1

G0[5,2] = -1
G0[5,4] = 1
G0[5,5] = -1

G0[6,3] = theta
G0[6,5] = -theta
G0[6,6] = 1
G0[6,7] = -1

G0[7,7] = 1

### G1 ####

G1[1,1] = 1

G1[3,1] = -c_bar
G1[3,4] = y_bar
G1[3,5] = (1-delta)*k_bar

G1[7,7] = gamma

Pi[1,1] = 1
Pi[1,2] = -beta*r_bar

Psi[7] = 1

sol2 = gensys(G0,G1,Psi,Pi)

irf1 = irf(sol1,100,0.01)
irf2 = irf(sol2,100,0.01)

plot(irf1[:,1])
plot!(irf2[:,1])

plot(irf1[:,2])
plot!(irf2[:,2])

plot(irf1[:,3])
plot!(irf2[:,3])

plot(irf1[:,4])
plot!(irf2[:,4])

plot(irf1[:,5])
plot!(irf2[:,5])

plot(irf1[:,6])
plot!(irf2[:,6])

G0 = zeros(6,6)
G1 = zeros(6,6)
Pi = zeros(6,2)
Psi = zeros(6)

G0[1,1] = 1
G0[1,2] = -r_bar*beta

G0[3,5] = 1

G0[6,6] = 1

###G1####

G1[1,1] = 1

G1[2,1] = 1
G1[2,3] = 1
G1[2,4] = -1

G1[3,1] = r_bar/theta-delta
G1[3,4] = r_bar/theta
G1[3,5] = (1-delta)

G1[4,3] = 1-theta
G1[4,4] = -1
G1[4,5] = theta
G1[4,6] = 1

G1[5,2] = -1
G1[5,4] = 1
G1[5,5] = -1

G1[6,6] = gamma

Pi[1,1] = 1
Pi[1,2] = -beta*r_bar

Psi[6] = sig

sol3 = gensys(G0,G1,Psi,Pi)
irf3 = irf(sol3,20,0.5)


plot(irf1[:,1])
plot!(irf2[:,1])
plot!(irf3[:,1])

#################

G0 = zeros(7,7)
G1 = zeros(7,7)
Pi = zeros(7,2)
Psi = zeros(7)

G0[1,1] = 1
G0[1,2] = -r_bar*beta

G0[2,1] = -1
G0[2,3] = -1
G0[2,4] = 1

G0[3,5] = 1

G0[4,3] = -(1-theta)
G0[4,4] = 1
G0[4,5] = -theta
G0[4,7] = -1

G0[5,2] = 1
G0[5,4] = -1
G0[5,5] = 1

G0[6,3] = 1
G0[6,4] = -1
G0[6,6] = 1

G0[7,7] = 1

##### G1 #####

G1[1,1] = 1

G1[3,1] = -(r_bar/theta-delta)
G1[3,4] = r_bar/theta
G1[3,5] = 1-delta

G1[7,7] = gamma

Pi[1,1] = 1
Pi[1,2] = -beta*r_bar

Psi[7] = 1

sol4 = gensys(G0,G1,Psi,Pi)

irf4 = irf(sol4,100,0.5)

plot(irf1[:,1])
plot!(irf4[:,1])

plot(irf1[:,2])
plot!(irf4[:,2])

plot(irf1[:,3])
plot!(irf4[:,3])

plot(irf1[:,4])
plot!(irf4[:,4])

plot(irf1[:,5])
plot!(irf4[:,5])

plot(irf1[:,6])
plot!(irf4[:,6])

plot(irf1[:,7])
plot!(irf4[:,7])
