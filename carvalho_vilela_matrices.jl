## This is a crude replication of Carvalho and Vilela (2015)
## By Daniel Coutinho and Rafaela

####### WARNING############
## NOT DEBUGGED AS MUCH AS IT SHOULD!
#################

### TODO

## 1. Pi and Psi matrices
##2. More comments

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

G0 = zeros(23,23)
G1 = zeros(23,23)
Pi = zeros(23,6)
Psi = zeros(23,8)

G0[1,1] = 1
G0[1,5] = (1-h)/sigma
G0[1,6] = (1-h)/sigma
G0[1,3] = -(1-h)/sigma

G1[1,1] = (1-h)
G1[1,2] = h
G1[1,6] = (1-h)/sigma

G1[2,7] = -1
G1[2,1] = 1-alfa
G1[2,8] = alfa*eta*(2-alfa)
G1[2,9] = alfa*eta
G1[2,10] = alfa

G0[3,8] = 1

G1[3,8] = 1
G1[3,15] = -1
G1[3,17] = 1

G1[4,11] = -1
G1[4,9] = 1
G1[4,8] = 1 -alfa

G0[5,11] = -1
G0[5,14] = -1

G1[5,11] = 1
G1[5,12] = 1
G1[5,13] = -1
G1[5,5] = -1

G0[6,15] = beta

G1[6,15] = (1+beta*delta_d)
G1[6,16] = beta*delta_d
G1[6,8] = -kappa_d*alfa
G1[6,7] = -kappa_d*phi
G1[6,1] = -kappa_d*sigma/(1-h)
G1[6,2] = kappa_d*sigma/(1-h)
G1[6,19] = kappa_d*(1+phi)

G0[7,17] = beta

G1[7,17] = (1+beta*delta_i)
G1[7,18] = delta_i
G1[7,9] = kappa_i

G0[8,8] = -alfa

G1[8,5] = -1
G1[8,15] = 1
G1[8,8] = alfa

G0[9,20] = 1
G0[9,8] = alfa

G1[9,20] = 1/beta
G1[9,9] = -alfa
G1[9,7] = 1
G1[9,1] = -1

G0[19,12] = 1
G0[10,21] = 1
G0[10,3] = -1

G1[10,20] = chi

G0[11,3] = 1
G1[11,3] = rho_it1
G1[11,4] = rho_it2
G1[11,5] = (1 - rho_it1 - rho_it2)*lambda_pi
G1[11,7] = (1 - rho_it1 - rho_it2)*lambda_y
G1[11,12] = (1 - rho_it1 - rho_it2)*lambda_s
G1[11,13] = -(1 - rho_it1 - rho_it2)*lambda_s

G0[12,10] = 1

G1[12,10] = a1_yy
G1[12,14] = a1_ypi
G1[12,21] = a1_yi

G0[13,14] = 1
G0[13,10] = a0_piy

G1[13,10] = a1_piy
G1[13,14] = a1_pipi
G1[13,21] = a1_pii

G0[14,21] = 1
G0[13,14] = ao_ipi
G0[13,10] = a0_iy

G1[14,10] = a1_iy
G1[14,14] = a1_ipi
G1[14,21] = a1_ii

G0[15,19] = 1

G1[15,19] = rho_a

G0[16,6] = 1

G1[16,6] = rho_gamma

G0[17,22] = 1

G1[17,22] = rho_varepsilon_cp

G0[18,23] = 1

G1[18,23] = rho_phi

G0[19,1] = 1
G1[19,2] = 1

G0[20,3] = 1
G1[20,4] = 1

G0[21,12] = 1
G1[21,13] = 1

G0[22,15] = 1
G1[22,16] = 1

G0[23,17] = 1
G1[23,18] = 1
