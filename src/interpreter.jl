using DynamicPolynomials

struct gensys_sys
    Gamma0::AbstractArray
    Gamma1::AbstractArray
    Psi::AbstractArray
    Pi::AbstractArray
end

function find_expectations(variables)
    vars_string = [repr(vars) for vars in variables]

    #check which variables are expectations and get their length

    expec_vars_test = match.(r"_e",vars_string)
    expec_vars_test = expec_vars_test .!= nothing
    length_expec = sum(expec_vars_test)
    println("Found ",length_expec, " expectations variables")
    expec_vars = variables[expec_vars_test]
    return expec_vars
end

function find_lags(variables)
    vars_string = [repr(vars) for vars in variables]

    #check which variables are expectations and get their length

    lag_vars_test = match.(r"_l",vars_string)
    lag_vars_test = lag_vars_test .!= nothing
    length_lag = sum(lag_vars_test)
    println("Found ",length_lag, " lags variables")
    lag_vars = variables[lag_vars_test]
    return lag_vars
end

function get_current(variables)
    vars_string = [repr(vars) for vars in variables]

    #check which variables are expectations and get their length

    cur_vars_test = match.(r"_",vars_string)
    cur_vars_test = cur_vars_test .=== nothing
    length_cur = sum(cur_vars_test)
    println("Found ",length_cur, " current variables")
    cur_vars = variables[cur_vars_test]
    return cur_vars
end

function get_current_from_lag(variables,lags)
    lags_string = [repr(vars) for vars in lags]
    vars_string = [repr(vars) for vars in variables]

    stripped_vars = [rstrip(vv,['_','l']) for vv in lags_string]

    current_vars_index = zeros(Int64,length(lags))

    for j in 1:length(stripped_vars)
        test = stripped_vars[j] .== vars_string
        if sum(test) == 0
            @error "Lag but no current"
        else
            current_vars_index[j] = findfirst(test)
        end
    end
    return variables[current_vars_index]
end

function get_current_from_foward(variables,fowards)
    fowards_string = [repr(vars) for vars in fowards]
    vars_string = [repr(vars) for vars in variables]

    stripped_vars = [rstrip(vv,['_','f']) for vv in fowards_string]

    current_vars_index = zeros(Int64,length(fowards))

    for j in 1:length(stripped_vars)
        test = stripped_vars[j] .== vars_string
        if sum(test) == 0
            @error "Expectation but no current"
        else
            current_vars_index[j] = findfirst(test)
        end
    end
    return variables[current_vars_index]
end

function declare_aux_variables(expec_vars)
    expec_vars_string = [repr(vars) for vars in expec_vars]
     #transform variables in strings
    stripped_vars = [rstrip(vv,['_','e']) for vv in expec_vars_string]
    foward_vars = string.(stripped_vars,"_f")
    expectation_errors = string.("nu_",stripped_vars)
    expec_vars_aux = string.(expec_vars_string,"_aux")
    length_expec = length(expec_vars)

    all_vars = [reshape(foward_vars,1,length_expec) reshape(expectation_errors,1,length_expec)]

    def = "@polyvar"

    for vars in all_vars
        def = join([def,vars]," ")
    end
    eval(Meta.parse(def))

    def_aux = string.(expec_vars_aux,"=",foward_vars,"+",expectation_errors)
    eval.(Meta.parse.(def_aux))
    return foward_vars,expectation_errors,expec_vars_aux
end

function model2gensys(model,shocks)
    keys_model = collect(keys(model))
    varis = Dict()

    for i in keys(model)
        merge!(varis,Dict(i=>variables(model[i])))
    end

    vari_all = copy(varis[keys_model[1]])

    for i in keys(varis)
        union!(vari_all,variables(model[i]))
    end

    expecs = find_expectations(vari_all)
    lags = find_lags(vari_all)
    f_vars, expec_errors, expecs_aux = declare_aux_variables(expecs)

    expecs_aux = eval.(Meta.parse.(expecs_aux))
    f_vars = eval.(Meta.parse.(f_vars))
    expec_errors = eval.(Meta.parse.(expec_errors))

    currents = get_current(vari_all)
    cur_w_lags = get_current_from_lag(vari_all,lags)
    cur_w_f = get_current_from_foward(vari_all,f_vars)
    only_cur = setdiff(currents,[cur_w_lags;cur_w_f;shocks])


    for j in 1:length(expecs)
        for i in keys(model)
            model[i] = subs(model[i],expecs[j] => expecs_aux[j])
        end
    end

    foward_variables = [f_vars;setdiff(currents,[cur_w_f; shocks; only_cur])]
    backward_variables = [setdiff(currents,[cur_w_lags; shocks;only_cur]); lags; only_cur]

    #Building each matrix

    Gamma1 = zeros(length(model),length(model))
    Gamma0 = zeros(length(model),length(model))
    Psi = zeros(length(model),length(shocks))
    Pi =  zeros(length(model),length(expec_errors))

    #Gamma 0

    for j in 1:length(foward_variables)
        for i in keys(model)
            index = findfirst(keys_model .== i)
            Gamma0[index,j] = -coefficient(model[i],foward_variables[j])
        end
    end

    #Gamma 1

    for j in 1:length(backward_variables)
        for i in keys(model)
            index = findfirst(keys_model .== i)
            Gamma1[index,j] = coefficient(model[i],backward_variables[j])
        end
    end

    #Psi

    for j in 1:length(shocks)
        for i in keys(model)
            index = findfirst(keys_model .== i)
            Psi[index,j] = coefficient(model[i],shocks[j])
        end
    end

    #Pi

    for j in 1:length(expec_errors)
        for i in keys(model)
            index = findfirst(keys_model .== i)
            Pi[index,j] = coefficient(model[i],expec_errors[j])
        end
    end
    return gensys_sys(Gamma0,Gamma1,Psi,Pi)
end
