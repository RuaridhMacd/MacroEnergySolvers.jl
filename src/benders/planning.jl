"""
    solve_planning_problem(m::Model, planning_variables::Vector{String})

Solves the planning (upper-level) problem in the Benders decomposition algorithm.

This function attempts to solve the planning problem and handles potential numerical issues,
particularly with negative capacities, by rounding small values to zero if needed.

# Arguments
- `m::Model`: The JuMP model representing the planning problem
- `planning_variables::Vector{String}`: Names of the variables of the planning problem

# Returns
A NamedTuple containing:
- `fixed_cost`: Fixed cost component of the solution 
- `values`: Dictionary mapping linking variable names to their optimal values

# Notes
Only an optimal feasible solve supplies a lower bound. Solver-specific recovery
may retry a continuous affine master after a numerical failure. Infeasibility
triggers conflict diagnostics when supported; other unresolved failures raise
`BendersPlanningSolveError` so the algorithm can retain an existing incumbent.
"""
function retry_planning_problem!(m::Model, planning_variables::Vector{String})
    termination_status(m) in (SLOW_PROGRESS, NUMERICAL_ERROR, DUAL_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) || return nothing
    # The simplex fallback is appropriate only for a continuous, affine master.
    any(v -> is_integer(v) || is_binary(v), all_variables(m)) && return nothing
    objective_function(m) isa AffExpr || return nothing
    all(F in (VariableRef, AffExpr) for (F, _) in list_of_constraint_types(m)) || return nothing
    variants = benders_planning_retry_variants(unsafe_backend(m))
    isempty(variants) && return nothing
    original_attributes = Dict{String,Any}()
    try
        for attributes in variants
            @warn "Retrying numerical master LP solve; accuracy limits unchanged" attributes
            for (key,val) in attributes
                haskey(original_attributes,key) || (original_attributes[key] = get_optimizer_attribute(m,key))
                set_optimizer_attribute(m,key,val)
            end
            optimize!(m)
            if termination_status(m) == OPTIMAL && solved_and_feasible(m)
                # Restoring raw optimizer attributes can invalidate JuMP's cached
                # result status. Capture the accepted solution before restoration.
                return process_planning_sol(m, planning_variables), objective_value(m)
            end
        end
        throw(BendersPlanningSolveError("Planning recovery failed; no new lower bound was accepted. " * subproblem_status_summary(m)))
    finally
        for (key,val) in original_attributes
            set_optimizer_attribute(m,key,val)
        end
    end
    return nothing
end

function solve_planning_problem(m::Model,planning_variables::Vector{String})
	
	
    optimize!(m)
    if !(termination_status(m) == OPTIMAL && solved_and_feasible(m))
        recovered = retry_planning_problem!(m, planning_variables)
        isnothing(recovered) || return recovered
    end

    if termination_status(m) == OPTIMAL && solved_and_feasible(m)
        planning_sol = process_planning_sol(m,planning_variables)
        LB = objective_value(m)
    else
        if termination_status(m) == INFEASIBLE
            try
                compute_conflict!(m)
                list_of_conflicting_constraints = ConstraintRef[]
                for (F, S) in list_of_constraint_types(m), con in all_constraints(m, F, S)
                    if get_attribute(con, MOI.ConstraintConflictStatus()) == MOI.IN_CONFLICT
                        push!(list_of_conflicting_constraints, con)
                    end
                end
                display(list_of_conflicting_constraints)
            catch err
                @warn "Planning conflict diagnostics unavailable" reason=sprint(showerror, err)
            end
        end
        message = "Planning problem did not return an optimal feasible solution; no new lower bound was accepted. " * subproblem_status_summary(m)
        termination_status(m) == INFEASIBLE && error(message)
        throw(BendersPlanningSolveError(message))
    end

    return planning_sol, LB
end

function add_approximate_variable_cost!(m::Model, number_of_subproblems::Int)

    if haskey(m, :vTHETA)
        
        @warn "Variable vTHETA already exists in the model. Skipping addition of approximate variable cost because it has already been added."

        unregister(m, :ePlanningCost)

        @expression(m, ePlanningCost, objective_function(m) - sum(m[:vTHETA]))

        drop_zeros!(m[:ePlanningCost])

        return
    end

    @variable(m, vTHETA[w in 1:number_of_subproblems])

    if haskey(m, :eLowerBoundOperatingCost)
        @constraint(m, cThetaLowerBound[w in 1:number_of_subproblems], vTHETA[w] >= m[:eLowerBoundOperatingCost][w])
    else
        @constraint(m, cThetaLowerBound[w in 1:number_of_subproblems], vTHETA[w] >= 0.0)
    end

    @expression(m, ePlanningCost, objective_function(m))

    @objective(m, Min, m[:ePlanningCost] + sum(m[:vTHETA][w] for w in 1:number_of_subproblems))

end


function process_planning_sol(m::Model,planning_variables::Vector{String})

    capacity_variables = values(m[:eAvailableCapacity])

    if any(value(vcap) < -1e-8 for vcap in capacity_variables)
        @info "Found negative capacity values, setting them to zero."
        planning_variables_values = Dict();
        all_planning_variables = all_variables(m)
        for v in all_planning_variables
            if in(v,capacity_variables)
                planning_variables_values[v] = round_small_values(value(v))
            else
                planning_variables_values[v] = value(v)
            end
        end
        planning_cost = value(x->planning_variables_values[x], m[:ePlanningCost])
        planning_variables_values = Dict([s => planning_variables_values[variable_by_name(m,s)] for s in planning_variables])
    else
        planning_cost =  value(m[:ePlanningCost])
        planning_variables_values = Dict([s=>value.(variable_by_name(m,s)) for s in planning_variables])
    end

    planning_sol =  (planning_cost = planning_cost, values = planning_variables_values)

    return planning_sol
    
end

function round_small_values(z::Float64)
    if z < -1e-8
        return 0.0
    else
        return z
    end
end
