module MacroEnergySolversMosekExt

using MacroEnergySolvers, Mosek, MosekTools

# A single retry, not a sequence of increasingly permissive tolerances.
MacroEnergySolvers.benders_phase_one_retry_scale(::MosekTools.Optimizer) = 100.0
MacroEnergySolvers.benders_operation_retry_targets(::MosekTools.Optimizer) = (1000.0, 100.0, 10000.0)
MacroEnergySolvers.benders_operation_retry_variants(::MosekTools.Optimizer) = (
    ("MSK_IPAR_INTPNT_SOLVE_FORM"=>Mosek.MSK_SOLVE_PRIMAL.value,),
    ("MSK_IPAR_INTPNT_SOLVE_FORM"=>Mosek.MSK_SOLVE_DUAL.value,),
)
MacroEnergySolvers.benders_planning_retry_variants(::MosekTools.Optimizer) = (
    ("MSK_IPAR_INTPNT_SOLVE_FORM"=>Mosek.MSK_SOLVE_DUAL.value,),
    ("MSK_IPAR_OPTIMIZER"=>Mosek.MSK_OPTIMIZER_DUAL_SIMPLEX.value,),
)

function MacroEnergySolvers.benders_native_infeasible(optimizer::MosekTools.Optimizer)
    return Mosek.solutiondef(optimizer.task, Mosek.MSK_SOL_ITR) &&
           Mosek.getsolsta(optimizer.task, Mosek.MSK_SOL_ITR) == Mosek.MSK_SOL_STA_PRIM_INFEAS_CER
end

function MacroEnergySolvers.benders_solution_quality(optimizer::MosekTools.Optimizer)
    task = optimizer.task
    Mosek.solutiondef(task, Mosek.MSK_SOL_ITR) || return nothing
    r = Mosek.getsolutioninfonew(task, Mosek.MSK_SOL_ITR)
    # Include affine conic constraint violations (ACC), not only legacy cones.
    return (
        native_optimal=Mosek.getsolsta(task, Mosek.MSK_SOL_ITR) == Mosek.MSK_SOL_STA_OPTIMAL,
        primal_objective=r[1], dual_objective=r[9],
        primal_violation=maximum(r[2:8]), dual_violation=maximum(r[10:14]),
    )
end

end
