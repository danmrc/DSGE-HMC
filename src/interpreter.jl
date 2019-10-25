using DynamicPolynomials

macro declare_variables(endogenous,shocks)
     #transform variables in strings
    endo_string = [repr(vars) for vars in endogenous]
    shocks_string = [repr(vars) for vars in shocks]

    #check which variables are expectations and get their length

    expec_vars_test = match.(r"_e",endo_string)
    expec_vars_test = expec_vars_test .!= nothing
    length_expec =

end
