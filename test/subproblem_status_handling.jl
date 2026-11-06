# Mock statuses reproduce QCP failure modes deterministically, without requiring
# a Gurobi license or depending on a particular numerical failure occurring.
@testset "Benders subproblem status handling" begin
    MOI = JuMP.MOI
    @testset "unreliable results never enter feasibility recovery" begin
        for (status, primal, dual_result) in (
            (MOI.NUMERICAL_ERROR, MOI.NO_SOLUTION, MOI.NO_SOLUTION),
            (MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT, MOI.NO_SOLUTION),
            (MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT),
            (MOI.OPTIMAL, MOI.FEASIBLE_POINT, MOI.NO_SOLUTION),
            (MOI.INFEASIBLE_OR_UNBOUNDED, MOI.NO_SOLUTION, MOI.NO_SOLUTION),
            (MOI.TIME_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT),
            (MOI.SLOW_PROGRESS, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT),
            (MOI.SLOW_PROGRESS, MOI.NO_SOLUTION, MOI.INFEASIBILITY_CERTIFICATE),
        ), expect_feasible in (false, true)
            mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
            calls = Ref(0)
            MOI.Utilities.set_mock_optimize!(mock, optimizer -> begin
                calls[] += 1
                MOI.Utilities.mock_optimize!(optimizer, status, (primal, [1.0]), dual_result)
            end)
            m = direct_model(mock)
            @variable(m, x)
            @objective(m, Min, x)
            err = try
                MES.solve_subproblem(m, (values=Dict("x" => 1.0),), ["x"], expect_feasible)
            catch e
                e
            end
            @test err isa MES.BendersSubproblemSolveError
            @test occursin("infeasibility has not been established", sprint(showerror, err))
            @test occursin("termination=$status", sprint(showerror, err))
            @test calls[] == 1
        end
    end

    @testset "actual linking duals must be finite" begin
        mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}();
            eval_variable_constraint_dual=false)
        MOI.Utilities.set_mock_optimize!(mock, optimizer ->
            MOI.Utilities.mock_optimize!(optimizer, MOI.OPTIMAL,
                (MOI.FEASIBLE_POINT, [1.0]), MOI.FEASIBLE_POINT,
                (MOI.VariableIndex, MOI.EqualTo{Float64}) => [NaN]))
        m = direct_model(mock)
        @variable(m, x)
        @objective(m, Min, x)
        @test_throws r"non-finite fixing dual" MES.solve_subproblem(
            m, (values=Dict("x" => 1.0),), ["x"], false)
    end

    @testset "dual-status flag alone does not guarantee retrievable duals" begin
        mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}();
            eval_variable_constraint_dual=false)
        MOI.Utilities.set_mock_optimize!(mock, optimizer ->
            MOI.Utilities.mock_optimize!(optimizer, MOI.OPTIMAL,
                (MOI.FEASIBLE_POINT, [1.0]), MOI.FEASIBLE_POINT))
        m = direct_model(mock)
        @variable(m, x)
        @objective(m, Min, x)
        @test_throws r"unable to retrieve the fixing dual" MES.solve_subproblem(
            m, (values=Dict("x" => 1.0),), ["x"], false)
    end

    @testset "failed feasibility solve restores the model" begin
        m = Model(HiGHS.Optimizer)
        set_silent(m)
        @variable(m, x)
        @variable(m, y >= 5)
        set_upper_bound(y, 3.0)
        @constraint(m, y <= x)
        @objective(m, Min, 2y + 1)
        MES.add_slacks_to_subproblem!(m)
        original = objective_function(m)
        @test_throws r"Feasibility subproblem" MES.solve_subproblem(
            m, (values=Dict("x" => 0.0),), ["x"], false)
        @test is_fixed(m[:slack_max])
        @test fix_value(m[:slack_max]) == 0.0
        @test objective_sense(m) == MOI.MIN_SENSE
        @test isequal(objective_function(m), original)
        @test !haskey(m.ext, :MacroEnergySolversPhaseOneScale)
    end

    @testset "zero phase-I objective does not generate a feasibility cut" begin
        mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
        MOI.Utilities.set_mock_optimize!(mock,
            optimizer -> MOI.Utilities.mock_optimize!(optimizer, MOI.INFEASIBLE),
            optimizer -> MOI.Utilities.mock_optimize!(optimizer, MOI.OPTIMAL,
                (MOI.FEASIBLE_POINT, [0.0]), MOI.FEASIBLE_POINT))
        m = direct_model(mock)
        @variable(m, slack_max == 0)
        @objective(m, Min, 2slack_max + 1)
        sol = MES.solve_subproblem(m, (values=Dict{String,Float64}(),), String[], false)
        @test sol.theta_coeff == 1
        @test sol.op_cost == 1.0
        @test fix_value(slack_max) == 0.0
        @test coefficient(objective_function(m), slack_max) == 2.0
        @test !haskey(m.ext, :MacroEnergySolversOperationScale)
        @test !haskey(m.ext, :MacroEnergySolversPhaseOneScale)
    end

    @testset "zero phase I is not by itself a usable operating solution" begin
        mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
        MOI.Utilities.set_mock_optimize!(mock,
            o -> MOI.Utilities.mock_optimize!(o, MOI.INFEASIBLE),
            o -> MOI.Utilities.mock_optimize!(o, MOI.OPTIMAL,
                (MOI.FEASIBLE_POINT,[0.0]), MOI.FEASIBLE_POINT),
            o -> MOI.Utilities.mock_optimize!(o, MOI.NUMERICAL_ERROR))
        m = direct_model(mock)
        @variable(m, slack_max == 0)
        @objective(m, Min, 2slack_max + 1)
        @test_throws r"Operating retry after near-zero phase I" MES.solve_subproblem(
            m, (values=Dict{String,Float64}(),), String[], false)
        @test fix_value(slack_max) == 0
        @test coefficient(objective_function(m),slack_max) == 2
    end

    @testset "numerical failure during phase I restores the model" begin
        mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
        MOI.Utilities.set_mock_optimize!(mock,
            optimizer -> MOI.Utilities.mock_optimize!(optimizer, MOI.INFEASIBLE),
            optimizer -> MOI.Utilities.mock_optimize!(optimizer, MOI.NUMERICAL_ERROR))
        m = direct_model(mock)
        @variable(m, slack_max == 0)
        @objective(m, Min, 2slack_max + 1)
        @test_throws r"Feasibility subproblem did not return an optimal primal/dual solution" MES.solve_subproblem(
            m, (values=Dict{String,Float64}(),), String[], false)
        @test fix_value(slack_max) == 0.0
        @test coefficient(objective_function(m), slack_max) == 2.0
    end
end

@testset "Row-relative primal residuals" begin
    mock = MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
    MOI.Utilities.set_mock_optimize!(mock, o -> MOI.Utilities.mock_optimize!(o,
        MOI.OPTIMAL, (MOI.FEASIBLE_POINT,[1e6+0.1]), MOI.FEASIBLE_POINT))
    m = direct_model(mock)
    @variable(m,x)
    @constraint(m,c,x<=1e6)
    @objective(m,Min,x)
    optimize!(m)
    residual = MES.relative_primal_violation(m)
    @test residual < 1e-6
    @test residual > 1e-8
    set_normalized_coefficient(c,x,100.0)
    set_normalized_rhs(c,1e8)
    optimize!(m)
    @test MES.relative_primal_violation(m) ≈ residual
    set_normalized_rhs(c,1.0)
    optimize!(m)
    @test MES.relative_primal_violation(m) > 0.9
end

@testset "Safeguarded stalled solutions" begin
    quality = (native_optimal=true, primal_objective=0.15, dual_objective=0.15-1e-9,
               primal_violation=1e-10, dual_violation=7e-7)
    @test MES.acceptable_stalled_quality(quality)
    scaled_quality = merge(quality, (primal_objective=100*quality.primal_objective,
        dual_objective=100*quality.dual_objective, dual_violation=100*quality.dual_violation))
    normalized = MES.normalize_benders_quality(scaled_quality, 100.0)
    @test MES.acceptable_stalled_quality(normalized)
    @test normalized.primal_violation == quality.primal_violation
    @test normalized.dual_violation ≈ quality.dual_violation
    @test normalized.dual_objective ≈ quality.dual_objective
    @test !MES.acceptable_stalled_quality(scaled_quality)
    @test !MES.acceptable_stalled_quality(nothing)
    for bad in ((native_optimal=false,), (primal_violation=2e-6,),
                (dual_violation=2e-6,), (dual_objective=0.1,),
                (primal_objective=NaN,), (dual_violation=Inf,))
        @test !MES.acceptable_stalled_quality(merge(quality, bad))
    end

    # Cuts use the dual intercept, while upper bounds keep the primal cost.
    m = Model()
    @variable(m, x)
    @variable(m, vTHETA[1:1])
    plan = (values=Dict("x"=>0.0), planning_cost=2.0)
    sol = Dict(1 => (op_cost=10.0, cut_cost=9.0, lambda=[-1.0], theta_coeff=1))
    MES.update_planning_problem_multi_cuts!(m, sol, plan, Dict(1=>["x"]))
    c = only(all_constraints(m, AffExpr, MOI.GreaterThan{Float64}))
    @test normalized_rhs(c) == 9.0
    @test MES.compute_upper_bound(m, plan, sol) == 12.0
end

# A distinct mock backend opts into the same recovery capability as MOSEK.
# Certificates from failed economic solves are never used directly for cuts.
const RecoveryMock = typeof(MOI.Utilities.MockOptimizer(
    MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}());
    eval_variable_constraint_dual=false))
MES.benders_operation_retry_targets(::RecoveryMock) = (10.0,)
MES.benders_planning_retry_variants(::RecoveryMock) = (("test_recovery_attribute"=>1,),)
MOI.get(::RecoveryMock, ::MOI.RawOptimizerAttribute) = 0
MOI.set(::RecoveryMock, ::MOI.RawOptimizerAttribute, ::Any) = nothing
function MES.benders_solution_quality(o::RecoveryMock)
    obj = MOI.get(o, MOI.ObjectiveValue())
    return (native_optimal=MOI.get(o,MOI.TerminationStatus()) == MOI.OPTIMAL,
        primal_objective=obj, dual_objective=obj,
        primal_violation=0.0, dual_violation=0.0)
end

@testset "Master lower bounds require optimality" begin
    calls=Ref(0)
    mock=MOI.Utilities.MockOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()))
    MOI.Utilities.set_mock_optimize!(mock, o -> begin
        calls[]+=1
        MOI.Utilities.mock_optimize!(o, calls[]==1 ? MOI.SLOW_PROGRESS : MOI.OPTIMAL,
            (MOI.FEASIBLE_POINT,[2.0]), MOI.FEASIBLE_POINT)
    end)
    m=Model(() -> mock) # attribute restoration invalidates this JuMP cache
    @variable(m,x>=0)
    @objective(m,Min,100x)
    m[:eAvailableCapacity]=Dict(1=>x)
    m[:ePlanningCost]=100x
    sol,lb=MES.solve_planning_problem(m,["x"])
    @test calls[]==2
    @test lb≈200.0
    @test sol.values["x"]≈2.0

    for status in (MOI.SLOW_PROGRESS,MOI.TIME_LIMIT)
        mock=MOI.Utilities.MockOptimizer(MOI.Utilities.Model{Float64}())
        MOI.Utilities.set_mock_optimize!(mock,o->MOI.Utilities.mock_optimize!(o,status,
            (MOI.FEASIBLE_POINT,[2.0]),MOI.FEASIBLE_POINT))
        m=direct_model(mock)
        @variable(m,x>=0)
        @objective(m,Min,100x)
        m[:eAvailableCapacity]=Dict(1=>x)
        @test_throws MES.BendersPlanningSolveError MES.solve_planning_problem(m,["x"])
    end
end

@testset "Recover numerical economic certificates" begin
    for initial_status in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED, MOI.NUMERICAL_ERROR)
        mock = MOI.Utilities.MockOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}());
            eval_variable_constraint_dual=false)
        calls = Ref(0)
        MOI.Utilities.set_mock_optimize!(mock, o -> begin
            calls[] += 1
            if calls[] == 1
                MOI.Utilities.mock_optimize!(o,initial_status)
            else
                MOI.Utilities.mock_optimize!(o, MOI.OPTIMAL,
                    (MOI.FEASIBLE_POINT,[2.0]), MOI.FEASIBLE_POINT,
                    (MOI.VariableIndex,MOI.EqualTo{Float64}) => [10.0])
            end
        end)
        m = direct_model(mock)
        @variable(m,x)
        @objective(m,Min,100x)
        sol = MES.solve_subproblem(m,(values=Dict("x"=>2.0),),["x"],true)
        @test calls[] == 2
        @test sol.theta_coeff == 1
        @test sol.op_cost ≈ 200.0
        @test sol.lambda ≈ [100.0]
        @test coefficient(objective_function(m),x) == 100.0
        @test m.ext[:MacroEnergySolversLastObjectiveScale] ≈ 0.1
        @test !haskey(m.ext,:MacroEnergySolversOperationScale)
    end
    mock = MOI.Utilities.MockOptimizer(MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}());
        eval_variable_constraint_dual=false)
    MOI.Utilities.set_mock_optimize!(mock,o->MOI.Utilities.mock_optimize!(o,MOI.DUAL_INFEASIBLE))
    m = direct_model(mock)
    @variable(m,x)
    @objective(m,Min,100x)
    @test_throws MES.BendersSubproblemSolveError MES.solve_subproblem(m,
        (values=Dict("x"=>2.0),),["x"],true)
    @test coefficient(objective_function(m),x) == 100.0
end
