using AdvancedHMC
using Distributions, Random
using ForwardDiff, StatsPlots

include(string(pwd(),"/src/simulation.jl"))
include(string(pwd(),"/gali_bayesian.jl"))
include(string(pwd(),"/mcmc/priors_v2.jl"))
include(string(pwd(),"/mcmc/renormalization_manual.jl"))

yy,shocks = simulate_dsge(GAMMA_0,GAMMA_1,PSI,PI,500)

#ll = dens([0,0,0,0,0,0,0,0,0,0],yy[:,2])
#ll2,grd = dens_and_grad([0,0,0,0,0,0,0,0,0,0],yy[:,2])

ell(par) = dens([1/3;par],yy[:,2])
ell_grad(par) = dens_and_grad([1/3;par],yy[:,2])

from_unit(x) = -log(1/x-1)
from_pos(x) = log(x)
from_one_to_inf(x) = log(x-1)

aa = [from_unit(0.99),from_pos(6),from_unit(2/3),from_pos(1),from_pos(1),from_pos(1),from_one_to_inf(1.5),from_pos(0.5/4),from_unit(0.5)]

ll,dd = ell_grad(aa)

m = inv(diagm(diag(dd*dd'))/500)

n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(9)
hamiltonian = Hamiltonian(metric, ell,ell_grad)

initial = aa

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian,initial)
integrator = JitteredLeapfrog(initial_ϵ,1.0)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = StaticTrajectory(integrator,100)#NUTS{MultinomialTS,ClassicNoUTurn}(integrator) #
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.65, integrator))#StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial, n_samples, adaptor, n_adapts; progress=true)

tab = zeros(2000,9)

for j = 1:size(samples,1)
    tab[j,:] = samples[j]
end

true_pars  = [0.99,6,2/3,1,1,1,1.5,0.5/4,0.5]

StatsPlots.density(to_unit.(tab[:,1]))
vline!([true_pars[1]])
StatsPlots.density(to_positive.(tab[:,2]))
vline!([true_pars[2]])
StatsPlots.density(to_unit.(tab[:,3]))
vline!([true_pars[3]])
StatsPlots.density(to_positive.(tab[:,4]))
vline!([true_pars[4]])
StatsPlots.density(to_positive.(tab[:,5]))
vline!([true_pars[5]])
StatsPlots.density(to_positive.(tab[:,6]))
vline!([true_pars[6]])
StatsPlots.density(to_one_inf.(tab[:,7]))
vline!([true_pars[7]])
StatsPlots.density(to_positive.(tab[:,8]))
vline!([true_pars[8]])
StatsPlots.density(to_unit.(tab[:,9]))
vline!([true_pars[9]])
