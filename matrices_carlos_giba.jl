# matrices of Gensys
G0 = zeros(24,24)
G1 = zeros(24,24)
Psi = zeros(24,8)
Pi = zeros(24,6)

# Equation 1 (Euler)

G0[1,1] = -1
G0[1,2] = -(1-h)/sig
G0[1,3] = (1-h)/sig
G0[1,7] = 1+h
G0[1,8] = (1-h)/sig

G1[1,3] = (1+h)/sig
G1[1,7] = h

Pi[1,1] = -1
Pi[1,2] = (1-h)/sig
Pi[1,3] = (1-h)/sig

# Equation 2 (goods market equilibrium)

G0[2,9]  = 1
G0[2,7]  = -(1-alfa)
G0[2,10] = -alfa*eta*(2-alfa)
G0[2,11] = -alfa*eta
G0[2,17] = -alfa

# Equation 3 (tot)

G0[3,10] = 1
G0[3,13] = -1
G0[3,12] = 1

G1[3,10] = 1

# Equation 4 (relationship q and tot)

G0[4,10] = -(1-alfa)
G0[4,14] = 1
G0[4,11] = -1

# Equation 5 (relationship q, s, pi)

G0[5,14] = 1
G0[5,15] = -1
G0[5,18] = -1

G1[5,2]  = -1
G1[5,14] = 1
G1[5,15] = -1

# Equation 6 (phillips curve - domestic)

G0[6,5]  = -bet
G0[6,12] = 1-bet*delta_D
G0[6,24] = bet*(1-theta_D)*(1-theta_D*bet)/theta_D

G1[6,12] = delta_D

Pi[6,4] = -bet

# Equation 7 (marginal cost)

G0[7,24] = 1
G0[7,10] = -alfa
G0[7,9]  = -phi
G0[7,7]  = -sig/(1-h)
G0[7,20] = 1+phi

G1[7,7] = -(sig*h)/(1-h)

# Equation 8  (phillips curve - importing)

G0[8,6]  = -bet
G0[8,13] = 1+bet*delta_I
G0[8,11] = -(1-theta_I)*(1-theta_I*bet)/theta_I
G0[8,22] = -1

G1[8,13] = delta_I

Pi[8,5]  = -bet

# Equation 9 (relationship pi and tot)

G0[9,12] = -1
G0[9,10] = -alfa

G1[9,10] = alfa
G1[9,2]  = -1

# Equation 10 (bugdet constraint)

G0[10,21] = 1
G0[10,10] = alfa
G0[10,11] = alfa
G0[10,9]  = -1
G0[10,7]  = 1

G1[10,21] = 1/bet

# Equation 11 (UIP)

G0[11,4]  = -1
G0[11,15] = 1
G0[11,8]  = 1
G0[11,19] = -1
G0[11,21] = chi
G0[11,23] = -1

Pi[11,6] = -1

# Equation 12 (Taylor RMI)

G0[12,8]  = 1
G0[12,16] = -rho_1
G0[12,9]  = -(1-rho_1-rho_2)*lambda_y
G0[12,15] = -(1-rho_1-rho_2)*lambda_s

G1[12,16] = rho_2
G1[12,2]  = (1-rho_1-rho_2)*lambda_pi
G1[12,15] = -(1-rho_1-rho_2)*lambda_s

Psi[12,5] = sigma_i

###################
## Foreign Block ##
###################

# System (Equations 13, 14 and 15)

G0[13,17] = 1
G0[14,17] = a_0piy
G0[14,18] = 1
G0[15,17] = a_0iy
G0[15,18] = a_0ipi
G0[15,19] = 1

G1[13,17] = a_1yy
G1[13,18] = a_1ypi
G1[13,19] = a_1yi
G1[14,17] = a_1piy
G1[14,18] = a_1pipi
G1[14,19] = a_1pii
G1[15,17] = a_1iy
G1[15,18] = a_1ipi
G1[15,19] = a_1ii

Psi[13,6] = sigma_y_star
Psi[14,7] = sigma_pi_star
Psi[15,8] = sigma_i_star

#################
## Shocks block #
#################

# Equation 16 (a)

G0[16,20] = 1

G1[16,20] = rho_a

Psi[16,1] = sigma_a

# Equation 17 (gamma)

G0[17,3] = 1

G1[17,3] = rho_gamma

Psi[17,2] = sigma_gamma

# Equation 18 (epsilon_cp)

G0[18,22] = 1

G1[18,22] = rho_cp

Psi[18,3] = sigma_cp

# Equation 19 (phi)

G0[19,23] = 1

G1[19,23] = rho_phi

Psi[19,4] = sigma_phi


####################
## Identity block ##
####################

# Equation 20 (c)

G0[20,7] = 1
G1[20,1] = 1

# Equation 21 (s)

G0[21,15] = 1
G1[21,4]  = 1

# Equation 22 (pi_D)

G0[22,12] = 1
G1[22,5]  = 1

# Equation 23 (pi_I)

G0[23,13] = 1
G1[23,6]  = 1

# Equation 24 (i)

G0[24,16] = 1
G1[24,8]  = 1

# G0[findall(G0 .== nothing)] .= 0
# G1[findall(G1 .== nothing)] .= 0
# Psi[findall(Psi .== nothing)] .= 0
# Pi[findall(Pi .== nothing)] .= 0
