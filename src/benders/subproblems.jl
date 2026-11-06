
function add_slacks_to_subproblems!(m_subproblems::Vector{Dict{Any, Any}})
    
    add_slacks_to_local_subproblems!(m_subproblems); 

    return nothing
end

function add_slacks_to_subproblems!(m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}})

    @sync for p in workers()
        @async @spawnat p begin
            add_slacks_to_local_subproblems!(localpart(m_subproblems));
        end
    end

    return nothing
end

function add_slacks_to_local_subproblems!(subproblem_local::Vector{Dict{Any,Any}})

    for sp in subproblem_local
        add_slacks_to_subproblem!(sp[:model]);
    end
    return nothing
end


function add_slacks_to_subproblem!(subproblem::Model)
    ### Slack variables are added to the subproblems and fixed to zero. 
    ### We will then allow slack variables to be non-zero to generate feasibility cuts when a subproblem is infeasible.

    if haskey(subproblem, :slack_max)
        @warn "Variable slack_max already exists in the model. Skipping addition of slack variables"
        @warn "If user is rerunning a model and has updated any constraints, new constraints added since last solve will NOT include the slack. Proceed with caution."
        return nothing
    end

    eq_cons =  all_constraints(subproblem,AffExpr,MOI.EqualTo{Float64})
    less_ineq_cons = all_constraints(subproblem,AffExpr,MOI.LessThan{Float64})
    greater_ineq_cons = all_constraints(subproblem,AffExpr,MOI.GreaterThan{Float64})


    @variable(subproblem, slack_max)
    @constraint(subproblem, slack_max >= 0)

    if !isempty(less_ineq_cons)
        for c in less_ineq_cons
            set_normalized_coefficient(c, slack_max, -1)
        end
    end

    if !isempty(greater_ineq_cons)
        for c in greater_ineq_cons
            set_normalized_coefficient(c, slack_max, 1)
        end
    end

    if !isempty(eq_cons)
        n = length(eq_cons)
        @variable(subproblem, slack_eq[1:n])
        for i in 1:n
            set_normalized_coefficient(eq_cons[i], slack_eq[i], -1)
        end
        @constraint(subproblem, [i in 1:n], slack_eq[i] <= slack_max)
        @constraint(subproblem, [i in 1:n], -slack_eq[i] <= slack_max)
    end

        fix.(slack_max,0.0); 
    

    return nothing
end

function fix_linking_variables!(m::Model,planning_sol::NamedTuple,linking_variables_sub::Vector{String})
    ### Fix linking variables in the subproblem to the values computed by the planning problem. 
	for y in linking_variables_sub
		vy = variable_by_name(m,y);
		fix(vy,planning_sol.values[y];force=true)
		if is_integer(vy)
			unset_integer(vy)
		elseif is_binary(vy)
			unset_binary(vy)
		end
	end
end

function recover_operation_after_zero_slack!(m::Model, linking::Vector{String}, objective;
        resolve::Bool=true, context::String="Operating retry after near-zero phase I")
    if resolve
        @warn "Phase I found near-zero violation; retrying the operating objective instead of generating a feasibility cut"
        optimize!(m)
    end
    retry_key = :MacroEnergySolversOperationScale
    original_attributes = Dict{String,Any}()
    try
        if !solved_for_benders_cut(m)
            targets = benders_operation_retry_targets(unsafe_backend(m))
            if !isnothing(targets) && objective isa AffExpr
                magnitude = maximum((abs(c) for (c,_) in linear_terms(objective)); init=0.0)
                for attributes in benders_operation_retry_variants(unsafe_backend(m))
                    for (key,val) in attributes
                        haskey(original_attributes,key) || (original_attributes[key]=get_optimizer_attribute(m,key))
                        set_optimizer_attribute(m,key,val)
                    end
                    for target in targets
                        retry_scale = target/max(1.0,magnitude)
                        @warn "Retrying operating objective in normalized cost units" retry_scale target magnitude attributes
                        m.ext[retry_key] = retry_scale
                        set_objective_function(m, retry_scale*objective)
                        optimize!(m)
                        solved_for_benders_cut(m) && break
                    end
                    solved_for_benders_cut(m) && break
                end
            end
        end
        cost, lambda = benders_cut_values(m, linking, context)
        return (op_cost=cost, lambda=lambda, theta_coeff=1, cut_cost=benders_cut_intercept(m,cost))
    finally
        for (key,val) in original_attributes
            set_optimizer_attribute(m,key,val)
        end
        delete!(m.ext, retry_key)
    end
end

function solve_subproblem(m::Model,planning_sol::NamedTuple,linking_variables_sub::Vector{String},expect_feasible_subproblems::Bool; feasibility_tolerance::Float64=1e-7)
	
    ### Solve the operational subproblem. If it is infeasible, compute feasibility cuts.

    isfinite(feasibility_tolerance) && feasibility_tolerance >= 0 ||
        throw(ArgumentError("feasibility_tolerance must be finite and nonnegative"))
	fix_linking_variables!(m,planning_sol,linking_variables_sub)

	optimize!(m)
	
    if solved_for_benders_cut(m)
        op_cost, lambda = benders_cut_values(m, linking_variables_sub, "Operational subproblem")
        return (op_cost=op_cost, lambda=lambda, theta_coeff=1,
                cut_cost=benders_cut_intercept(m, op_cost))
    end

    if termination_status(m) in (SLOW_PROGRESS, NUMERICAL_ERROR, DUAL_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED) &&
       !subproblem_reported_infeasible(m) &&
       !isnothing(benders_operation_retry_targets(unsafe_backend(m)))
        original_objective = objective_function(m)
        try
            return recover_operation_after_zero_slack!(m, linking_variables_sub, original_objective;
                resolve=false, context="Operational retry")
        finally
            set_objective_function(m, original_objective)
        end
    end

    # A numerical failure, missing dual solution, or ambiguous infeasibility status
    # does not certify infeasibility. Never build a cut from such a result.
    if !subproblem_reported_infeasible(m)
        throw(BendersSubproblemSolveError("Operational subproblem did not produce a usable optimal solution; " *
              "infeasibility has not been established and no Benders cut was generated. " *
              subproblem_status_summary(m)))
    end
    if expect_feasible_subproblems
        error("Operational subproblem is infeasible, but ExpectFeasibleSubproblems = true. " *
              "Set it to false to generate feasibility cuts. " * subproblem_status_summary(m))
    end
    haskey(m, :slack_max) || error(
        "Operational subproblem is infeasible, but feasibility slacks have not been initialized.",
    )

    if termination_status(m) == SLOW_PROGRESS
        @warn "MOSEK returned a native infeasibility certificate despite stalling; attempting phase I only"
    end
    @info "Subproblem is infeasible, solving feasibility subproblem..."
    objfun = objective_function(m)
    objsense = objective_sense(m)
    scale_key = :MacroEnergySolversPhaseOneScale
    previous_scale = get(m.ext, scale_key, nothing)
    try
        m.ext[scale_key] = 1.0
        unfix(m[:slack_max])
        @objective(m, Min, m[:slack_max])
        optimize!(m)
        if !solved_for_benders_cut(m) && termination_status(m) == SLOW_PROGRESS
            retry_scale = benders_phase_one_retry_scale(unsafe_backend(m))
            if !isnothing(retry_scale)
                @warn "Retrying stalled MOSEK phase I with a scaled objective; accuracy limits unchanged" retry_scale quality=stalled_benders_quality(m)
                m.ext[scale_key] = retry_scale
                @objective(m, Min, retry_scale*m[:slack_max])
                optimize!(m)
            end
        end
        op_cost, lambda = benders_cut_values(m, linking_variables_sub, "Feasibility subproblem")
        cut_cost = benders_cut_intercept(m, op_cost)
        # A near-zero phase-I objective cannot meaningfully separate this plan.
        # This is an absolute threshold in the feasibility objective's units.
        if cut_cost <= feasibility_tolerance
            fix(m[:slack_max], 0.0; force=true)
            delete!(m.ext, scale_key)
            set_objective_sense(m, objsense)
            set_objective_function(m, objfun)
            return recover_operation_after_zero_slack!(m, linking_variables_sub, objfun)
        end
        return (op_cost=op_cost, lambda=lambda, theta_coeff=0, cut_cost=cut_cost)
    finally
        if isnothing(previous_scale)
            delete!(m.ext, scale_key)
        else
            m.ext[scale_key] = previous_scale
        end
        fix(m[:slack_max], 0.0; force=true)
        set_objective_sense(m, objsense)
        set_objective_function(m, objfun)
    end

end


function solve_local_subproblems(subproblem_local::Vector{Dict{Any,Any}},planning_sol::NamedTuple, expect_feasible_subproblems::Bool)

    local_sol=Dict();
    for sp in subproblem_local
        m = sp[:model];
        linking_variables_sub = sp[:linking_variables_sub]
        w = sp[:subproblem_index];
        try
            local_sol[w] = solve_subproblem(m,planning_sol,linking_variables_sub,expect_feasible_subproblems)
        catch
            @error "Benders subproblem failed; no result returned for cut generation" subproblem_index=w
            rethrow()
        end
    end
    return local_sol
end

"""
    solve_subproblems(
        m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}}, 
        planning_sol::NamedTuple
    )

Solves subproblems in parallel using distributed computing capabilities.

This function coordinates the parallel solution of operational subproblems across multiple workers,
using Julia's distributed computing framework. Each worker processes its local portion of the
distributed array of subproblems.

# Arguments
- `m_subproblems::DArray`: Distributed array containing the subproblems, where each element is a
   dictionary representing a subproblem
- `planning_sol::NamedTuple`: Current solution of the planning problem containing variable values
   needed for the subproblem solutions

# Returns
A merged dictionary containing results from all subproblems, where each entry contains:
- Optimal objective value
- Dual variables
- Other solution information from each subproblem

# Implementation Details
Uses `@sync` and `@async` for coordinated parallel execution, with results fetched from each worker
and merged into a single dictionary containing all subproblem solutions.
"""
function solve_subproblems(m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}},planning_sol::NamedTuple,expect_feasible_subproblems::Bool)

    p_id = workers();
    np_id = length(p_id);

    sub_results = [Dict() for k in 1:np_id];

    @sync for k in 1:np_id
              @async sub_results[k]= @fetchfrom p_id[k] solve_local_subproblems(localpart(m_subproblems),planning_sol,expect_feasible_subproblems); ### This is equivalent to fetch(@spawnat p .....)
    end

	sub_results = merge(sub_results...);

    return sub_results
end


function solve_subproblems(m_subproblems::Vector{Dict{Any, Any}},planning_sol::NamedTuple,expect_feasible_subproblems::Bool)
    
    sub_results = solve_local_subproblems(m_subproblems,planning_sol,expect_feasible_subproblems); 

    return sub_results
end
