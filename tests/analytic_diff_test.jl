using QuantEcon

include(string(pwd(),"/hmc/diffs_estrut.jl"))
include(string(pwd(),"/hmc/diff_kalman.jl"))
include(string(pwd(),"/src/simulation.jl"))

aa = [1/3,0.99,6,2/3,1,1,1,1.5,0.5/4,0.5]

teste_diff_mod = diff_mod(aa,1)

yy,shocks = simulate_dsge(GAMMA_0_foo(aa),GAMMA_1_foo(aa),PSI_foo(aa),PI_foo(aa),500)

data = yy[:,1]

sol = gensys(GAMMA_0_foo(aa),GAMMA_1_foo(aa),PSI_foo(aa),PI_foo(aa); verbose = false)

G = zeros(1,4)
G[1,2] = 1

A = sol.Theta1
R = [0] .+ 1e-8
Q = aa[6]^2*sol.Theta2*sol.Theta2'

dA = teste_diff_mod[:,1:16]
dG = teste_diff_mod[:,17:20]
dQ = teste_diff_mod[:,21:30]
dR = zeros(size(aa,1),1)

dx0 = zeros(4)
S0 = zeros(4,4)

g_x_hat_l = zeros(size(aa,1),4)
d_Sl = zeros(size(aa,1),size(S0,1)^2)

kalman_res = Kalman(A,G,Q,R) #create a Kalman filter instance

y_mean = mean(data,dims=1)
y_var = var(data,dims=1)
y_var = reshape(y_var,size(y_var,2))
#y_var = repeat([1],p)

x_hat = repeat(y_mean,4) #initial mean of the state
x_var = diagm(repeat(y_var,4))#variance initial of state
#x_var = x_var*x_var'
set_state!(kalman_res,x_hat,x_var)

med = kalman_res.cur_x_hat
varian = kalman_res.cur_sigma
#println(det(varian))
eta = data[2,:] - kalman_res.G*med #mean loglike
P = kalman_res.G*varian*kalman_res.G' + kalman_res.R
teste_cond = 1/cond(P)

gain = (A*varian*G')*inv(P)

d_p = diff_P(G,varian,dG,d_Sl,dR)
d_gain = diff_K(A,P,G,varian,dA,dG,d_p,d_Sl)
g_eta = grad_y(med,g_x_hat_l,G,dG)
g_x_hat = grad_x_hat(med,dA,A,g_x_hat_l,eta,g_eta, gain,d_gain)

p_inv = inv(P)

-1/2*vec(p_inv)'*d_p' - eta'*p_inv*g_eta' -1/2*(kron(eta'*p_inv, eta*p_inv)*d_p') #updating the kalman estimates

m = size(P,1)
n = size(A,1)
Knn = commutation_matrix(n,n)

b1 = (kron(A*varian,I(n)) + kron(I(n),A*varian)*Knn)*dA'
b2 = (kron(gain*P,I(n)) + kron(I(n),gain*P)*Kmn)*d_gain'
b1 + kron(A,A)*d_Sl' -b2 - kron(gain,gain)*d_p' + duplication_matrix(4)*dQ'

d_Sl = diff_S(A,P,gain,varian,dA,d_p,d_gain,d_Sl,dQ)
