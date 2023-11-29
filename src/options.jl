# Define options for the solvers. Some options will be ignored by some solvers,
# i.e. the GPU version will not check for convergence and will only run a certain
# number of iterations, where as the other solvers will ignore the iters option.

mutable struct Options
    # Tolerances
    eps_primal::Float64         # Tolerance for primal convergence
    eps_dual::Float64           # Tolerance for dual convergence

    # Penalty
    σ::Float64                  # Initial σ value (penalty on x, acts like regularizer for H in OSQP)
    ρ::Float64                  # Initial ρ value (penalty on constraints)
    ρ_min::Float64              # Max ρ
    ρ_max::Float64              # Min ρ
    adaptive_ρ::Bool            # Toggle ρ updates
    adaptive_ρ_spacing::Int     # Spacing between ρ updates

    # Relaxation param
    α::Float64                  # Relaxation param, should be between 0 and 1

    # Equality detection
    eq_tol::Float64             # Tolerance on when a constraint is an equality constraint

    # Solver settings
    max_iters::Integer          # Max number of iters
    verbose::Bool               # Flag for solver/setup output
    scaling::Bool               # Flag for problem scaling
    check_convergence::Bool     # Enable/disable convergence checking
    iters_btw_checks::Int       # Iterations between convergence checks
    warmstart::Bool             # Enable warmstarting (start with last soln)
end

function Options(; 
    eps_primal=1e-4, 
    eps_dual=1e-4, 
    σ=1e-06, 
    ρ=0.1, 
    ρ_min=1e-6, 
    ρ_max=1e+6, 
    adaptive_ρ=true, 
    adaptive_ρ_spacing=5, 
    α=1.6, 
    eq_tol=1e-6, 
    max_iters=4000, 
    verbose=false, 
    scaling=true,
    check_convergence=true,
    iters_btw_checks=25,
    warmstarting = true
    )
    Options(
        eps_primal, 
        eps_dual, 
        σ, 
        ρ, 
        ρ_min, 
        ρ_max, 
        adaptive_ρ, 
        adaptive_ρ_spacing, 
        α, 
        eq_tol, 
        max_iters, 
        verbose, 
        scaling,
        check_convergence,
        iters_btw_checks,
        warmstarting
    )
end
