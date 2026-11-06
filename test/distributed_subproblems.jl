using Distributed, DistributedArrays

@testset "Distributed numerical failure retains the incumbent" begin
    created=addprocs(2; exeflags=["--project=$(dirname(Base.active_project()))","--threads=1"])
    subproblems=nothing
    try
        owner=first(created)
        # The second worker deliberately owns no part of this Benders model.
        Distributed.remotecall_eval(Main,owner,:(using JuMP, HiGHS, MacroEnergySolvers, DistributedArrays))
        subproblems=distribute([Dict{Any,Any}()];procs=[owner])
        @fetchfrom owner begin
            s=Model(HiGHS.Optimizer);set_silent(s)
            @variable(s,x)
            @variable(s,g>=0)
            @constraint(s,g<=x)
            @constraint(s,g>=1)
            @objective(s,Min,2g)
            calls=Ref(0)
            set_optimize_hook(s,function(model;kwargs...)
                calls[]+=1
                calls[]>1 && throw(MacroEnergySolvers.BendersSubproblemSolveError("controlled distributed failure"))
                optimize!(model;ignore_optimize_hook=true,kwargs...)
            end)
            sp=only(localpart(subproblems))
            sp[:model]=s
            sp[:linking_variables_sub]=["x"]
            sp[:subproblem_index]=1
        end
        p=Model(HiGHS.Optimizer);set_silent(p)
        @variable(p,1<=x<=10)
        @objective(p,Min,x)
        p[:eAvailableCapacity]=Dict(1=>x)
        setup=Dict{Symbol,Any}(:MaxIter=>2,:MaxCpuTime=>60,:ConvTol=>0.01,
            :StabParam=>0.0,:StabDynamic=>false,:IntegerInvestment=>false,
            :ExpectFeasibleSubproblems=>false,:InitialPlanningSolution=>Dict("x"=>2.0))
        raw=MES.benders(p,subproblems,Dict(1=>["x"]),setup)
        @test raw.termination_status=="SUBPROBLEM_NUMERICAL_FAILURE"
        @test last(raw.UB_hist)≈4.0
        @test last(raw.LB_hist)≈3.0
        @test raw.planning_sol.values["x"]≈2.0
        @test raw.failure_message=="controlled distributed failure"
        @test last(created) in workers()
    finally
        isnothing(subproblems)||close(subproblems)
        rmprocs(created)
    end
end
