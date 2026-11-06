# MacroEnergySolvers.jl

This repository is designed to collect solution algorithms for macro-energy system optimization models. In the future, we hope to support several algorithms that can be applied to solve these large-scale mixed integer linear programs. 

Currently, the repository includes the regularized Benders decomposition algorithm described in:

_F. Pecci and J. D. Jenkins (2025), "Regularized Benders Decomposition for High Performance Capacity Expansion Models," in IEEE Transactions on Power Systems, doi: [10.1109/TPWRS.2025.3526413](https://ieeexplore.ieee.org/document/10829583)_.

Other algorithms are possible and could be added in the future (e.g., [Nested Benders Decomposition](https://www.sciencedirect.com/science/article/abs/pii/S0377221718304466)). If you would like to develop a decomposition method for macro-energy system models, please open a pull request or get in touch with us, we welcome any contributions.

Operational subproblems must return `OPTIMAL`, feasible primal and dual results,
and finite fixing duals for all linking variables before an optimality cut is
accepted. When Mosek and MosekTools are loaded, an optional extension also permits
`SLOW_PROGRESS` if MOSEK's native interior solution is optimal, both primal and
dual results are feasible, residual checks are finite and at most `1e-6`, and
the absolute primal-dual gap is at most
`1e-8 * (1 + max(abs(primal_objective), abs(dual_objective)))`.
Phase-I residuals retain absolute limits in original slack units, including
native affine cone residuals. Economic solves use original-model primal distances
divided by each row's activity/bound scale, and native dual residuals divided by
`1 +` the infinity norm of the linear objective's coefficients in original cost
units. This avoids treating dollar-valued stationarity residuals as slack units.
Unsupported nonlinear economic objectives still reject stalled results.
Such acceptance is logged. Its cut intercept uses the smaller of the primal and
dual objectives; upper bounds still use the primal operating cost. These are
numerical acceptance tolerances, not an exact arithmetic certificate.
A numerical error, other suboptimal termination, missing dual result, or
ambiguous `INFEASIBLE_OR_UNBOUNDED` status cannot supply a cut or bound;
unrecovered failures stop the run with status diagnostics. For a stalled MOSEK phase-I solve
that fails the quality gate, one retry multiplies only the slack objective by
`100`. Returned objective values, fixing duals, dual residuals, and gaps are
converted back to the original slack units before acceptance and cut generation.
The retry never loosens the quality limits, changes physical constraints, or
rescales economic objectives. Its temporary scale and objective are restored on exit.
If phase I finds violation below the feasibility-cut tolerance, the algorithm
re-solves the original operating objective with slacks fixed to zero. MOSEK may
then try both native primal and dual formulations with up to three cost-only
normalizations each, whose largest linear objective coefficient is approximately
`1000`, `100`, or `10000`. Original optimizer attributes are restored afterward;
returned objectives and fixing
duals are converted to original economic units, and the same relative quality
checks apply. Near-zero slack alone never supplies an incumbent or an optimality
cut: an accepted operating primal/dual result is still required.

The MOSEK operating retries also apply to `NUMERICAL_ERROR`, `DUAL_INFEASIBLE`,
and `INFEASIBLE_OR_UNBOUNDED` economic solves, as well as uncertified stalls.
Retries solve the same physical model; the original certificate is never used
for a cut. Genuine unresolved unboundedness therefore still fails the gate.

A continuous affine MOSEK master that stalls or returns an ambiguous numerical
status retries the dual interior formulation, then dual simplex. Attributes and
tolerances are restored afterward. Only an optimal feasible master solution
supplies a new lower bound. If master recovery fails after an incumbent exists,
the result has status `MASTER_NUMERICAL_FAILURE` and retains the last valid
lower bound and best feasible upper bound. It does not claim convergence.

The optional `setup[:InitialPlanningSolution]` dictionary supplies starting
planning values by variable name. The candidate must satisfy the initial master
constraints to absolute tolerance `1e-5`. The ordinary unrestricted master is
still solved for the lower bound; the supplied plan is evaluated by the ordinary
operating subproblem workflow before it can establish an upper bound. Its values
are never fixed in subsequent master iterations.

`setup[:GapDenominatorFloor]` defaults to `1.0` and must be finite and positive.
The relative gap is `abs(UB-LB)/max(abs(UB), abs(LB), GapDenominatorFloor)`.
Callers expressing objectives in different economic units must convert this
floor too. MacroEnergy does this automatically for its `ObjectiveScale` setting;
MESolvers otherwise receives and returns costs and bounds in the supplied units.

A later `BendersSubproblemSolveError` returns the last verified incumbent and
bounds with status `SUBPROBLEM_NUMERICAL_FAILURE` if an incumbent exists. The
failed candidate supplies no cut or upper bound. Without an incumbent the error
still propagates; programming/configuration exceptions are not swallowed.
Distributed solves preserve this numerical-failure classification when returning
from a worker, so incumbent retention is the same as in sequential execution.
Distributed solving and feasibility-slack initialization target the model's
owning processes rather than every Julia worker in the session.
An `INFEASIBLE` operational solve triggers feasibility recovery (when
`ExpectFeasibleSubproblems=false`). The MOSEK extension also permits recovery
after `SLOW_PROGRESS` if both the native solution status and JuMP's dual status
report a primal infeasibility certificate. That certificate is not used to build
a cut directly; the separate phase-I solve must pass the cut-quality gate.
Recovery must be optimal (or pass the safeguarded MOSEK stall checks) with usable
duals and a positive objective above `1e-7`, in the feasibility objective's units.
The internal `solve_subproblem` function accepts a `feasibility_tolerance` keyword
to override this threshold. Its original objective and zero-fixed feasibility
slack are restored even if recovery fails. Conic constraints and variable bounds
are not relaxed by the existing affine feasibility-slack machinery.

In the case of Benders decomposition, the user must generate the inputs for function `benders(planning_problem,subproblems,linking_variables_sub,setup)` where:

1. `planning_problem` is a JuMP model describing a planning problem (without auxiliary variables to estimate operating costs). The planning problem must include:
    - A JuMP expression named `eLowerBoundOperatingCost` representing an initial global under-estimator of the operating cost in each subperiod (default is `0.0`).
2. `subproblems` can be either a vector or a distributed array of `Dict`. Each dict has keys:
    - `:model =>` JuMP model of the subproblem (`Model`)
    - `:linking_variables_sub =>` linking variables that belong to the subproblem (`Vector{String}`)
    - `:subproblem_index =>` index of the subproblem as it appears in the array of subproblems (`Int64`)
3. `linking_variables_sub` is a `Dict` mapping each subproblem index to the corresponding `Vector{String}` of linking variables.
4. `setup` is a `Dict` containing the settings of the Benders run, whose defaults are:
    ```
        Dict(
        :MaxIter=> 50,
        :MaxCpuTime => 7200,
        :ConvTol => 1e-3,
        :StabParam => 0.0,
        :StabDynamic => false,
        :IntegerInvestment => false,
        :Distributed => false
    )
    ```
