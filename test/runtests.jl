using MacroEnergySolvers
using Test
using Aqua
using JuMP
using HiGHS

const MES = MacroEnergySolvers

@testset "MacroEnergySolvers.jl" begin
    @testset "MacroEnergySolvers.jl" begin
    end
    include("infeasible_solve_detection.jl")
    include("subproblem_status_handling.jl")
    include("initial_planning_solution.jl")
    include("distributed_subproblems.jl")
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MacroEnergySolvers)
    end
end
