### JuMP's has_values() reports only that the solver returned a primal point, not that the model
### was solved to feasibility: after an infeasible solve the point handed back is an
### INFEASIBLE_POINT, which is not NO_SOLUTION, so has_values() stays true. Once a model has been
### solved successfully even once, every later infeasible solve on it still looks solved. The
### helpers below check the solution statuses directly instead.

struct BendersSubproblemSolveError <: Exception
    message::String
end
Base.showerror(io::IO, err::BendersSubproblemSolveError) = print(io,err.message)

struct BendersPlanningSolveError <: Exception
    message::String
end
Base.showerror(io::IO, err::BendersPlanningSolveError) = print(io,err.message)

"""
    solved_and_feasible(m::Model)

Return `true` when the last solve of `m` produced a feasible primal point.

Unlike `has_values`, this is `false` after an infeasible solve, including an infeasible solve that
follows a successful one on the same model.
"""
function solved_and_feasible(m::Model)
    return primal_status(m) == FEASIBLE_POINT
end

"""
    solved_with_duals(m::Model)

Return `true` when the last solve of `m` produced a feasible primal point and a feasible dual
point.

Use this wherever duals are read. Note that it is always `false` for a model whose integrality
constraints are still switched on, since duals are not available for a mixed-integer problem.
"""
function solved_with_duals(m::Model)
    return solved_and_feasible(m) && dual_status(m) == FEASIBLE_POINT
end

# Solver-specific extensions may provide native solution-quality information.
# The default never permits a stalled solve to supply cuts.
benders_solution_quality(optimizer) = nothing
benders_native_infeasible(optimizer) = false
benders_phase_one_retry_scale(optimizer) = nothing
benders_operation_retry_targets(optimizer) = nothing
benders_operation_retry_variants(optimizer) = ((),)
benders_planning_retry_variants(optimizer) = ()

phase_one_objective_scale(m::Model) = get(m.ext, :MacroEnergySolversOperationScale,
    get(m.ext, :MacroEnergySolversPhaseOneScale, 1.0))

function normalize_benders_quality(q, scale::Float64)
    isnothing(q) && return nothing
    return merge(q, (
        primal_objective=q.primal_objective/scale,
        dual_objective=q.dual_objective/scale,
        dual_violation=q.dual_violation/scale,
    ))
end

row_activity_scale(f::VariableRef) = max(1.0, abs(value(f)))
row_activity_scale(f::AffExpr) = max(1.0,
    abs(constant(f)) + sum(abs(c * value(v)) for (c, v) in linear_terms(f); init=0.0))
row_activity_scale(f::AbstractVector) = max(1.0, sqrt(sum(abs2, value.(f))))
row_activity_scale(f) = 1.0

function row_residual_scale(c)
    obj = constraint_object(c)
    scale = row_activity_scale(obj.func)
    for field in (:value, :lower, :upper)
        hasproperty(obj.set, field) || continue
        bound = getproperty(obj.set, field)
        bound isa Real && isfinite(bound) && (scale = max(scale, abs(bound)))
    end
    return scale
end

function relative_primal_violation(m::Model)
    violations = primal_feasibility_report(m; atol=0.0)
    return maximum((violation / row_residual_scale(c) for (c, violation) in violations); init=0.0)
end

function benders_model_quality(m::Model)
    scale = phase_one_objective_scale(m)
    q = normalize_benders_quality(benders_solution_quality(unsafe_backend(m)), scale)
    isnothing(q) && return nothing
    # Phase I keeps the existing absolute limits in original slack units.
    if haskey(m.ext, :MacroEnergySolversPhaseOneScale) &&
       haskey(m, :slack_max) && !is_fixed(m[:slack_max])
        return q
    end
    # Economic dual residuals have cost units. Normalize by the objective's
    # coefficient norm in original cost units, rather than comparing dollars
    # directly with the absolute phase-I tolerance.
    f = objective_function(m)
    cost_norm = if f isa AffExpr
        maximum((abs(c)/scale for (c, _) in linear_terms(f)); init=0.0)
    elseif f isa VariableRef
        1.0/scale
    else
        return merge(q, (; dual_violation=Inf))  # Unsupported objective: reject stalls.
    end
    return merge(q, (; absolute_primal_violation=q.primal_violation,
        absolute_dual_violation=q.dual_violation,
        primal_violation=relative_primal_violation(m),
        dual_violation=q.dual_violation/(1.0+cost_norm), cost_norm,
        residual_units="row-relative primal, objective-relative dual"))
end

function subproblem_reported_infeasible(m::Model)
    termination_status(m) == INFEASIBLE && return true
    # This only authorizes a phase-I solve, never a cut from the certificate.
    return termination_status(m) == SLOW_PROGRESS &&
           dual_status(m) == MOI.INFEASIBILITY_CERTIFICATE &&
           benders_native_infeasible(unsafe_backend(m))
end

function acceptable_stalled_quality(q)
    isnothing(q) && return false
    q.native_optimal || return false
    metrics = (q.primal_objective, q.dual_objective, q.primal_violation, q.dual_violation)
    all(isfinite, metrics) || return false
    0 <= q.primal_violation <= 1e-6 || return false
    0 <= q.dual_violation <= 1e-6 || return false
    gap_tolerance = 1e-8 * (1 + max(abs(q.primal_objective), abs(q.dual_objective)))
    return abs(q.primal_objective - q.dual_objective) <= gap_tolerance
end

function stalled_benders_quality(m::Model)
    termination_status(m) == SLOW_PROGRESS && solved_with_duals(m) || return nothing
    return benders_model_quality(m)
end

# Gurobi LOCALLY_SOLVED and generic SLOW_PROGRESS results remain rejected.
function solved_for_benders_cut(m::Model)
    solved_with_duals(m) || return false
    if phase_one_objective_scale(m) != 1.0
        termination_status(m) in (OPTIMAL, SLOW_PROGRESS) || return false
        return acceptable_stalled_quality(benders_model_quality(m))
    end
    termination_status(m) == OPTIMAL && return true
    return acceptable_stalled_quality(stalled_benders_quality(m))
end

# For accepted stalled solves, anchor cuts at the dual lower bound rather than
# the primal objective. Keep the primal objective separately for upper bounds.
function benders_cut_intercept(m::Model, primal_objective::Float64)
    q = phase_one_objective_scale(m) == 1.0 ? stalled_benders_quality(m) :
        benders_model_quality(m)
    return isnothing(q) ? primal_objective : min(primal_objective, q.dual_objective)
end

function subproblem_status_summary(m::Model)
    raw = try
        raw_status(m)
    catch
        "unavailable"
    end
    summary = "termination=$(termination_status(m)), primal=$(primal_status(m)), " *
              "dual=$(dual_status(m)), raw=$(repr(raw))"
    if termination_status(m) == SLOW_PROGRESS
        summary *= ", native_quality=$(repr(stalled_benders_quality(m)))"
    end
    return summary
end

function benders_cut_values(m::Model, linking_variables::Vector{String}, context::String)
    solved_for_benders_cut(m) || throw(BendersSubproblemSolveError(
        "$context did not return an optimal primal/dual solution; no Benders cut generated. " *
        subproblem_status_summary(m),
    ))
    if termination_status(m) == SLOW_PROGRESS
        @warn "Accepting stalled MOSEK solution after native optimality and quality checks" quality=stalled_benders_quality(m)
    end
    scale = phase_one_objective_scale(m)
    cost = objective_value(m)/scale
    isfinite(cost) || throw(BendersSubproblemSolveError("$context returned a non-finite objective; no Benders cut generated."))
    lambda = Float64[]
    for name in linking_variables
        v = variable_by_name(m, name)
        isnothing(v) && error("$context: linking variable $(repr(name)) was not found.")
        multiplier = try
            dual(FixRef(v))
        catch err
            throw(BendersSubproblemSolveError("$context: unable to retrieve the fixing dual for $(repr(name)); " *
                  "no Benders cut generated. $(subproblem_status_summary(m)). " *
                  sprint(showerror, err)))
        end
        isfinite(multiplier) || throw(BendersSubproblemSolveError(
            "$context: non-finite fixing dual for $(repr(name)); no Benders cut generated.",
        ))
        push!(lambda, multiplier/scale)
    end
    m.ext[:MacroEnergySolversLastObjectiveScale] = scale
    return cost, lambda
end
