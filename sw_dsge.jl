using DSGE
using ModelConstructors

# Initialize the model, calculates steady state

mod = SmetsWouters()
DSGE.init_model_indices!(mod)
init_parameters!(mod)
steadystate!(mod)

#Generate Gensys matrices

Γ0,Γ1,C,Ψ,Π = DSGE.eqcond(mod)

# Solves the model using Klein solver - does not work for some reason

DSGE.klein(mod)


pp = ModelConstructors.transform_to_model_space(mod.parameters,randn(49)) # transform the parameters from the real line to the constrained version
ModelConstructors.transform_to_real_line(mod.parameters) #does the opposite

DSGE.update!(mod,pp) #update the model with new parameters
