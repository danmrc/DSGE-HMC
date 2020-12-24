using JLD

include("autocorrelation.jl")
include("priors.jl")

vals = load("values.jld")

smp = vals["samples"]
stats = vals["stats"]

aa = autocor(smp,1:10)

using Plots

l = @layout([a b c d; e f g h i])

bar(1:10,[aa[1,1,1:10],aa[2,2,1:10],aa[3,3,1:10],aa[4,4,1:10],aa[5,5,1:10],aa[6,6,1:10],aa[7,7,1:10],aa[8,8,1:10],aa[9,9,1:10]],layout = l, label =  ["β" "ϵ" "θ" "σ" "σ_v" "ϕ" "ϕ_π" "ϕ_y" "ρ"],size = (800,500))
png("imgs/autocor.png")
bar(1:10,aa[2,2,1:10],layout = l)
bar(aa[3,3,1:10])
bar(aa[4,4,1:10])
bar(aa[5,5,1:10])
bar(aa[7,7,1:10])
bar(aa[8,8,1:10])
bar(aa[9,9,1:10])

using LinearAlgebra

autocor_diag = mapreduce(x->aa[x,x,:],hcat,1:9)
sum_auto = sum(autocor_diag,dims=1)
eff_sample = 2000 ./(1 .+2*sum_auto)

using StatsPlots

to_unit(x) = 1/(1+exp(-x))
to_positive(x) = exp(x)
to_one_inf(x) = exp(x) + 1

StatsPlots.density(to_unit.(smp[1:2000,4]))
plot!(0:0.1:1,pdf(prior_sig,0:0.1:1))
