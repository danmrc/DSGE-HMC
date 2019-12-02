## This is a crude replication of Carvalho and Vilela (2015)
## By Daniel Coutinho and Rafaela

####### WARNING############
## NOT DEBUGGED AS MUCH AS IT SHOULD!
#################

### TODO

## 1. Pi and Psi matrices
## 2. More comments
## 3. Coefficients as strings for latexify

#######

### ORDER
## c_t
##c_{t-1}
##i_{t-1}
##i_{t-2}
## pi_t
## gamma_t
## y_t
## tot_{t-1}
## y^{star}_t
## q_{t-1}
## s_t
## s_{t-1}
## pi^{star}_t
## pi_{D,t}
## pi_{D,t-1}
## pi_{I,t}
## pi_{I,t-1}
## a_t
## z_t
## i^{star}_t
## varepsilon_{cp,t}
## phi_t

#Creating Matrices

G0 = Array{Any}(nothing,23,23)
G1 = Array{Any}(nothing,23,23)
Pi = zeros(23,6)
Psi = zeros(23,8)

#First equation

G0[1,1] = 1
G0[1,5] = "(1-h)/sigma()"
G0[1,6] = "(1-h)/sigma()"
G0[1,3] = "-(1-h)/sigma()"

G1[1,1] = "(1-h)"
G1[1,2] = "h"
G1[1,6] = "(1-h)/sigma()"

#Second Equation

G1[2,7] = -1
G1[2,1] = "1-alpha()"
G1[2,8] = "alpha()*eta()*(2-alpha())"
G1[2,9] = "alpha()*eta()"
G1[2,10] = "alpha()"

#Third Equation

G0[3,8] = 1

G1[3,8] = 1
G1[3,15] = -1
G1[3,17] = 1

#Fourth Equation

G1[4,11] = -1
G1[4,9] = 1
G1[4,8] = "1 -alpha()"

#Fifth equation

G0[5,11] = -1
G0[5,14] = -1

G1[5,11] = 1
G1[5,12] = 1
G1[5,13] = -1
G1[5,5] = -1

#Equation number 6

G0[6,15] = "beta()"

G1[6,15] = "1+delta()_d"
G1[6,16] = "beta()delta()_d"
G1[6,8] = "-kappa()_d*alpha()"
G1[6,7] = "-kappa()_d*phi()"
G1[6,1] = "-kappa()_d*sigma()/(1-h)"
G1[6,2] = "kappa()_d*sigma()/(1-h)"
G1[6,19] = "kappa()_d*(1+phi())"

#Equation number 7

G0[7,17] = "beta()"

G1[7,17] = "(1+beta()*delta()_i)"
G1[7,18] = "delta()_i"
G1[7,9] = "kappa()_i"

#Equation number 8

G0[8,8] = "-alpha()"

G1[8,5] = -1
G1[8,15] = 1
G1[8,8] = "alpha()"

#Equation number 9

G0[9,20] = 1
G0[9,8] = "alpha()"

G1[9,20] = "1/beta()"
G1[9,9] = "-alpha()"
G1[9,7] = 1
G1[9,1] = -1

# Equation number 10

G0[10,12] = 1
G0[10,21] = 1
G0[10,3] = -1

G1[10,20] = "chi()"

# Equation number 11: Taylor rule

G0[11,3] = 1
G1[11,3] = "rho()_it1"
G1[11,4] = "rho()_it2"
G1[11,5] = "(1-rho()_it1-rho()_it2)*lambda()_pi"
G1[11,7] = "(1-rho()_it1-rho()_it2)*lambda()_y"
G1[11,12] = "(1-rho()_it1-rho()_it2)*lambda()_s"
G1[11,13] = "-(1-rho()_it1-rho()_it2)*lambda()_s" ## its for the delta tot

###################
## Foreign Block ##
###################

# Equation 12

G0[12,10] = 1

G1[12,10] = "a_1yy"
G1[12,14] = "a_1y*pi"
G1[12,21] = "a_1yi"

#Equation 13

G0[13,14] = 1
G0[13,10] = "a_0piy"

G1[13,10] = "a_1piy"
G1[13,14] = "a_1pipi"
G1[13,21] = "a_1pii"

#Equation 14

G0[14,21] = 1
G0[14,14] = "a_0ipi"
G0[14,10] = "a_0iy"

G1[14,10] = "a_1iy"
G1[14,14] = "a_1ipi"
G1[14,21] = "a_1ii"

#################
## Shocks block #
#################

# Equation 15

G0[15,19] = 1

G1[15,19] = "rho()_a"

# Equation 16

G0[16,6] = 1

G1[16,6] = "rho()_gamma"

# Equation 17

G0[17,22] = 1

G1[17,22] = "rho()_varepsilon_cp"

# Equation 18

G0[18,23] = 1

G1[18,23] = "rho()_phi"

####################
## Identity block ##
####################

# Equation 19

G0[19,1] = 1
G1[19,2] = 1

# Eqquation 20

G0[20,3] = 1
G1[20,4] = 1

# Equation 21

G0[21,12] = 1
G1[21,13] = 1

# Equation 22

G0[22,15] = 1
G1[22,16] = 1

# Equation 23

G0[23,17] = 1
G1[23,18] = 1

G0[findall(G0 .== nothing)] .= 0
G1[findall(G1 .== nothing)] .= 0

latexify(G0)
latexify(G1)
