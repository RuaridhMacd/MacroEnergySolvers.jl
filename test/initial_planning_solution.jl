@testset "Supplied initial planning solution" begin
    function seeded_example(initial; fail_after_first=false, fail_master=false)
        p=Model(HiGHS.Optimizer); set_silent(p)
        @variable(p,1 <= x <= 10)
        @objective(p,Min,x)
        p[:eAvailableCapacity]=Dict(1=>x)
        if fail_master
            calls=Ref(0)
            set_optimize_hook(p,function(model;kwargs...)
                calls[]+=1
                calls[]>1 && throw(MES.BendersPlanningSolveError("controlled master failure"))
                optimize!(model;ignore_optimize_hook=true,kwargs...)
            end)
        end
        s=Model(HiGHS.Optimizer); set_silent(s)
        @variable(s,x)
        @variable(s,g>=0)
        @constraint(s,g<=x)
        @constraint(s,g>=1)
        @objective(s,Min,2g)
        if fail_after_first
            calls=Ref(0)
            set_optimize_hook(s,function(model;kwargs...)
                calls[]+=1
                calls[]>1 && throw(MES.BendersSubproblemSolveError("controlled test failure"))
                optimize!(model;ignore_optimize_hook=true,kwargs...)
            end)
        end
        setup=Dict{Symbol,Any}(:MaxIter=>(fail_after_first || fail_master) ? 2 : 0,:MaxCpuTime=>10,:ConvTol=>0.01,
            :StabParam=>0.0,:StabDynamic=>false,:IntegerInvestment=>false,
            :ExpectFeasibleSubproblems=>false,:InitialPlanningSolution=>initial)
        return MES.benders(p,[Dict{Any,Any}(:model=>s,:linking_variables_sub=>["x"],
            :subproblem_index=>1)],Dict(1=>["x"]),setup)
    end
    raw=seeded_example(Dict("x"=>2.0))
    @test only(raw.LB_hist) ≈ 3.0  # Unrestricted master optimum, not the seeded cost.
    @test only(raw.UB_hist) ≈ 4.0  # The supplied feasible x=2 is actually evaluated.
    @test raw.planning_sol.values["x"] ≈ 2.0
    @test_throws r"violates master" seeded_example(Dict("x"=>20.0))
    @test_throws r"Unknown initial" seeded_example(Dict("unknown"=>2.0))
    stopped=seeded_example(Dict("x"=>2.0);fail_after_first=true)
    @test stopped.termination_status == "SUBPROBLEM_NUMERICAL_FAILURE"
    @test last(stopped.UB_hist) ≈ 4.0
    @test last(stopped.LB_hist) ≈ 3.0
    @test stopped.planning_sol.values["x"] ≈ 2.0
    @test stopped.failure_message == "controlled test failure"
    stopped_master=seeded_example(Dict("x"=>2.0);fail_master=true)
    @test stopped_master.termination_status == "MASTER_NUMERICAL_FAILURE"
    @test last(stopped_master.UB_hist) ≈ 4.0
    @test last(stopped_master.LB_hist) ≈ 1.0 # failed master supplies no replacement LB
    @test stopped_master.planning_sol.values["x"] ≈ 2.0
    @test stopped_master.failure_message == "controlled master failure"
end
