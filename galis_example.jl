### Gali's Book Chapter 3 model
### We use this as a consistency check of gensys implementation
### By Daniel Coutinho and Gilberto Boaretto

#### Parameters from Gali's book example, chapter 3, page 52, first edition

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

## Gali's give analytical expressions for the effects of a shock in the monetary policy over inflation and output gap,chapter 3, page 51, first edition

LAMBDA_v = ((1-bet*rho_v)*(sig*(1-rho_v)+phi_y)+kappa*(phi_pi-rho_v))^(-1)

y_irf(v) = -(1-bet*rho_v)*LAMBDA_v*v
pi_irf(v) = -kappa*LAMBDA_v*v
r_irf(v) = sig*(1-rho_v)*(1-bet*rho_v)*LAMBDA_v*v
i_irf(v) = (sig*(1-rho_v)*(1-bet*rho_v) - rho_v*kappa)*LAMBDA_v*v
v(v,uu) = rho_v*v + uu

## Test drive: lets see if this gives the same IRFs that we get in Gali's book
## This is just for debugging, so comment it out when using it to check gensys consistency

#using Plots

irfs = zeros(13,5)

irfs[1,1] = 0

uu = 0.25

for t in 2:13
    irfs[t,1] = v(irfs[t-1,1],uu)
    irfs[t,2] = pi_irf(irfs[t,1])
    irfs[t,3] = y_irf(irfs[t,1])
    irfs[t,4] = i_irf(irfs[t,1])
    irfs[t,5] = r_irf(irfs[t,1])
    global uu = 0
end

#plot(irfs[2:13,1], label = "v")
#plot(irfs[2:13,2]*4, label = "pi")
#plot(irfs[2:13,3], label = "y")
#plot(4*irfs[2:13,4], label = "i")
#plot(4*irfs[2:13,5], label = "r")
