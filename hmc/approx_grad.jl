using LogDensityProblems
using Distributions, Parameters, Random
using PositiveFactorizations, Calculus
using Flux

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/mcmc/priors_v2.jl"))
include(string(pwd(),"/mcmc/foos.jl"))

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

prob = DSGE_Model(yy[:,2],1/3)

prob((bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

t = problem_transform(prob)

P = TransformedLogDensity(t,prob)

aa = [0.99,6,2/3,1,1,1,1.5,0.5/4,0.5]

true_vals = TransformVariables.inverse(t,(bet = 0.99,epsilon = 6,theta=2/3,sig=1,s2=1,phi=1,phi_pi=1.5,phi_y=0.5/4,rho_v=0.5))

LogDensityProblems.logdensity(P,true_vals)

hes = -Calculus.hessian(x->LogDensityProblems.logdensity(P,x),true_vals)

hes_inv = inv(hes)

hes_inv = cholesky(Positive,hes_inv)

hes_inv = hes_inv.L*hes_inv.L'

isposdef(hes_inv)

dist = MvNormal(hes_inv)

N = 10000

dados = zeros(Float32,10,N)

i = 1

while i <= N
    x = rand(dist)
    y = LogDensityProblems.logdensity(P,x)
    if y < -90000
        continue
    else
        dados[1:9,i] = x
        dados[10,i] = y
        global i += 1
    end
    println(i)
end

dados_orig = dados
dados[10,:] = log.(-dados[10,:])
dados = Flux.normalise(dados)

model_flux = Chain(
    Dense(9,350,relu),
    Dense(350,500,relu),
    Dense(500,1)
)

function make_minibatch(X, Y, siz)
    indexes = sample(1:size(Y,1),siz)
    X_batch = X[:,indexes]
    Y_batch = Y[indexes]
    return (X_batch, Y_batch), indexes
end

mini_batch_test = [make_minibatch(dados_x,dados_y,100) for i in 1:100]

loss_flux(x,y) = Flux.mse(model_flux(x),y)

optim_flux = Flux.ADAM()

epochs_max = 100
batch_size = 350
patience = 5

acc(x,y) = mean([(model_flux(x)[1,i]-y[i])^2 for i in 1:batch_size])

loss = zeros(epochs_max)

final_countdown = 0
epoch = 2
last_update = 0

while epoch <= epochs_max&&final_countdown <= patience
    batch,id = make_minibatch(dados[1:9,:],dados[10,:],batch_size)
    batch = [batch for i in 1:batch_size]
    Flux.train!(loss_flux,Flux.params(model_flux),batch,optim_flux)
    loss_in = acc(dados[1:9,id],dados[10,id])*1000
    batch,id = make_minibatch(dados[1:9,:],dados[10,:],batch_size)
    loss[epoch] = acc(dados[1:9,id],dados[10,id])*1000
    println("Epoch ", epoch, " Loss out-of-sample *1000 ", loss[epoch]," Loss in-sample *1000 ", loss_in)
    delta_loss = (loss[epoch-1] - loss[epoch])/loss[epoch]
    if delta_loss < 1e-4
        if last_update != epoch-1
            global final_countdown = 0
            global last_update = epoch
        end
        global final_countdown += 1
    end
        global epoch += 1
end


idxs = sample(1:size(dados,2),1000)

scatter(reshape(model_flux(dados[1:9,idxs]),1000), label = "Fit")
scatter!(dados[10,idxs], label = "Data")
