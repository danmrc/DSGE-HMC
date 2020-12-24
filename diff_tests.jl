include("gensys.jl")
include("gali_bayesian.jl")
include("diffs_estrut_v3.jl")

using FiniteDifferences

bet = 0.99
sig = 1
phi = 1
alfa = 1/3
epsilon = 6
theta = 2/3
phi_pi = 1.5
phi_y = 0.5/4
rho_v = 0.5
s2 = 1

true_pars = [alfa,bet,epsilon,theta, sig, s2, phi,phi_pi, phi_y,rho_v]

##############################################################
#### DSGE diff
#################################################################

gen_sol = gensys(GAMMA_0_foo(true_pars),GAMMA_1_foo(true_pars),PSI_foo(true_pars),PI_foo(true_pars))

Th1(par) = vec(gensys(GAMMA_0_foo(par),GAMMA_1_foo(par),PSI_foo(par),PI_foo(par),verbose=false).Theta1)
Th2(par) = vec(gensys(GAMMA_0_foo(par),GAMMA_1_foo(par),PSI_foo(par),PI_foo(par),verbose=false).Theta2)

Th1(true_pars)

dA,dB = diff_mod(true_pars)

dAf = FiniteDifferences.jacobian(central_fdm(5,1),Th1,true_pars)
dBf = FiniteDifferences.jacobian(central_fdm(5,1),Th2,true_pars)

maximum(abs.(Array(dA) - dAf[1]))
maximum(abs.(Array(dB) - dBf[1]))

model1 = gensys(GAMMA_0_foo(true_pars),GAMMA_1_foo(true_pars),PSI_foo(true_pars),PI_foo(true_pars))

pert_pars = copy(true_pars)
pert_pars[1] = pert_pars[1] + 0.05

model2 = gensys(GAMMA_0_foo(pert_pars),GAMMA_1_foo(pert_pars),PSI_foo(pert_pars),PI_foo(pert_pars))

model1.Theta1 - model2.Theta1

###################################################
############### Kalman diff ####################
######################################################

using QuantEcon

function solve_lyapunov_vec(A,Q)
    aux = (I - kron(A,A))
    vec_ans = \(aux,vec(Q))
    res = reshape(vec_ans,size(Q))
    return res
end

A = 0.9
Q = 1
G = 1
R = 1

theta = 10
y = theta .+ randn(10)

kalman_res = Kalman(A,G,Q,R)

S0 = solve_lyapunov_vec(A,[Q])

set_state!(kalman_res,8.0,S0[1])

#update!(kalman_res,y[1])

P = kalman_res.G*kalman_res.cur_sigma*kalman_res.G' + kalman_res.R
gain = (kalman_res.A*kalman_res.cur_sigma*kalman_res.G')*inv(P)

A_per = 0.9 + 1e-5
S0_per = solve_lyapunov_vec(A_per,[Q])

kalman_res2 = Kalman(A_per,G,Q,R)

set_state!(kalman_res2,8.0,S0_per[1])

#update!(kalman_res2,y[1])

P2 = kalman_res2.G*kalman_res2.cur_sigma*kalman_res2.G' + kalman_res2.R
gain2 = (kalman_res2.A*kalman_res2.cur_sigma*kalman_res2.G')*inv(P2)

(P2 - P)/(1e-5)

(gain2 - gain)/(1e-5)

(kalman_res2.cur_sigma - kalman_res.cur_sigma)/(1e-5)

dff = (kalman_res2.cur_x_hat - kalman_res.cur_x_hat)/(1e-5)

ll = -1/2*logdet(P) - 1/2*(y[1] - G*kalman_res.cur_x_hat)'*inv(P)*(y[1] - G*kalman_res.cur_x_hat)
ll2 = -1/2*logdet(P2) - 1/2*(y[1] - G*kalman_res2.cur_x_hat)'*inv(P2)*(y[1] - G*kalman_res2.cur_x_hat)
dll = (ll2 - ll)/1e-5

(S0_per - S0)/1e-5

include("diff_kalman.jl")

S0 = solve_lyapunov_vec(A,[Q])
d_S0 = diff_S0(A,S0,1,[0])

d_mat = diff_mats([1],d_S0,[0],[0],[0])

dP = diff_P(G,kalman_res.cur_sigma,d_S0,[0])

d_gain = diff_K(A,P,G,kalman_res.cur_sigma,d_mat.dA,dP,d_mat.dS_l)

g_eta = grad_y(kalman_res.cur_x_hat,d_mat.dx_hat_l,kalman_res.G)

obs_err = y[1] - kalman_res.G*kalman_res.cur_x_hat

df1 = diff_kalman(kalman_res,d_mat,y[1])

update!(kalman_res,y[1])
update!(kalman_res2,y[1])

dff = (kalman_res2.cur_x_hat - kalman_res.cur_x_hat)/(1e-5)

g_med_state = grad_x_hat(kalman_res.cur_x_hat,kalman_res.A, gain, obs_err,d_mat.dx_hat_l, d_mat.dA, d_gain, g_eta)



##############################################
############## Diff llh final ################
#############################################
include("simulation.jl")
include("llh_diff.jl")

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

ll_foo(par) = loglike_dsge(par,yy[:,2])[1]

dif_finite = FiniteDifferences.grad(central_fdm(5,1),ll_foo,true_pars)

llh,dif = loglike_dsge(true_pars,yy[:,2])
