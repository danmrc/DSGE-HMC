using Flux

m = Flux.Chain(
    Dense(500,32,relu),
    Dense(32,1,relu)
)

loss(x,y) = Flux.mse(m(x),y)

ps = Flux.param(m)

x = rand(500)
y = rand(50)

opt = ADAM()

dat = [(x,y)]

Flux.train!(loss,ps,dat,opt)
