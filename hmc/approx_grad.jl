using LogDensityProblems, DynamicHMC
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
    Dense(9,20,relu),
    Dense(20,50,relu),
    Dense(50,50,relu),
    Dense(50,1)
)

function make_minibatch(X, Y, siz)
    indexes = sample(1:size(Y,1),siz)
    X_batch = X[:,indexes]
    Y_batch = Y[indexes]
    return (X_batch, Y_batch), indexes
end

#mini_batch_test = [make_minibatch(dados_x,dados_y,100) for i in 1:100]

loss_flux(x,y) = Flux.mse(model_flux(x),y)

optim_flux = Flux.RMSProp(0.00001)

epochs_max = 100
batch_size = 625
patience = 5

ofs_prop = 0.2

ofs_size = Int(ceil(batch_size*ofs_prop))

acc(x,y) = mean([(model_flux(x)[1,i]-y[i])^2 for i in 1:length(y)])

final_countdown = 0
epoch = 2
last_update = 0

while epoch <= epochs_max&&final_countdown <= patience
    batch,id = make_minibatch(dados[1:9,:],dados[10,:],batch_size)
    batch = [batch for i in 1:batch_size]

    train_batch = batch[1:(ofs_size)]

    train_batch_ids = id[1:(ofs_size)]
    val_batch_ids = id[(ofs_size+1):batch_size]

    loss_in_bef = acc(dados[1:9,train_batch_ids],dados[10,train_batch_ids])*1000
    loss_val_bef = acc(dados[1:9,val_batch_ids],dados[10,val_batch_ids])*1000

    Flux.train!(loss_flux,Flux.params(model_flux),train_batch,optim_flux)

    loss_in_aft = acc(dados[1:9,train_batch_ids],dados[10,train_batch_ids])*1000
    loss_val_aft = acc(dados[1:9,val_batch_ids],dados[10,val_batch_ids])*1000

    delta_loss_val = loss_val_aft - loss_val_bef
    delta_loss_in = loss_in_aft - loss_in_bef

    println("Epoch ", epoch, " ΔLoss out-of-sample *1000 ", delta_loss_val," ΔLoss in-sample *1000 ", delta_loss_in)

    if delta_loss_val > 1e-4
        if last_update != epoch-1
            global final_countdown = 0
        end
        global final_countdown += 1
        global last_update = epoch
    end
        global epoch += 1
end


idxs = sample(1:size(dados,2),500)

scatter(reshape(model_flux(dados[1:9,idxs]),500), label = "Fit")
scatter!(dados[10,idxs], label = "Data")

scatter(reshape(model_flux(dados[1:9,idxs]),500),dados[10,idxs])

acc(dados[1:9,idxs],dados[10,idxs])
