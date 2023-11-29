"""
Iteration layer for the solver on the GPU. Updates the input in-place as x ← Wx + b
then clamps the indices corresponding with the splitting variable z
"""
mutable struct SolverIterationLayer
    W::CuArray{Float64, 2}                      # Weight matrix
    b::CuArray{Float64, 1}                      # Bias vector
    l::CuArray{Float64, 1}                      # Lower bound on z
    u::CuArray{Float64, 1}                      # Upper bound on z
    z_inds::UnitRange{Int64}                    # z indices (for clamping)
end
Flux.@functor SolverIterationLayer

"""
Workspace for the solver. Stores the current solution (for warmstarting), views
into the solution for calculating residuals. Also stores the previous solution
ρ index, the norm of the cost gradient, and the primal and dual residuals.
"""
mutable struct Workspace
    gpu_soln::CuArray{Float64}                  # Storage for gpu solution (for warm-starting)
    x::CuArray{Float64}                         # View into gpu_soln for x
    z::CuArray{Float64}                         # View into gpu_soln for z
    λ::CuArray{Float64}                         # View into gpu_soln for λ
    prev_soln_ρ_ind::Int64                      # ρ from previous solution
    g_norm::Float64                             # Norm of the unscaled g for ρ updates
    primal_res::Float64                         # norm of primal residual
    dual_res::Float64                           # norm of dual residual
    primal_normalizer::Float64                  # Primal normalization for penalty updating
    dual_normalizer::Float64                    # Dual normalization for penalty updating
    J::Float64                                  # Stored cost, only used in verbose mode
    function Workspace(nx::Int64, nc::Int64, g::Vector{Float64})
        # Allocate solution (x, z, λ)
        gpu_soln = CUDA.zeros(Float64, nx + 2*nc)
        x = @view gpu_soln[1:nx]
        z = @view gpu_soln[nx .+ (1:nc)]
        λ = @view gpu_soln[nx + nc .+ (1:nc)]

        return new(gpu_soln, x, z, λ, 0, CUDA.norm(g), 0.0, 0.0, 0.0)
    end
end

"""
Stores the results from the solver on the CPU. Includes primal and dual solution,
splitting variable, cost, and estimated run time
"""
mutable struct Results
    x::Vector{Float64}                          # Primal variable
    z::Vector{Float64}                          # Constraint primal variable
    λ::Vector{Float64}                          # Dual variable
    J::Float64                                  # Cost
    solve_time::Float64                         # Solve time
    iters::Int64                                # Total iterations
    function Results(nx::Int64, nc::Int64)
        return new(zeros(nx), zeros(nc), zeros(nc), 0.0, 0.0, 0)
    end
end

"""
Data for the solver, including problem formulation on CPU and GPU (possibly scaled),
penalty values, layers and bias matrices. Also containts the options, solver workspace,
and solver results structs.
"""
mutable struct Solver
    prob::QPProb                                # Problem definition (CPU side)
    gpu_prob::GPU_QPProb                        # Problem definition (GPU side, possibly scaled)
    ρs::Vector{Float64}                         # Array of predefined ρ
    start_ρ_ind::Int64                          # ρ to start with if not warmstarting
    bias_mat_ρs::Vector{CuArray{Float64}}       # Update bias matrix, i.e. b = B*g
    layer_ρs::Vector{SolverIterationLayer}      # Update layers
    opts::Options                               # Solver options
    workspace::Workspace                        # Solver workspace
    results::Results                            # Solver results
    function Solver(opts, prob, gpu_prob)
        workspace = Workspace(prob.nx, prob.nc, prob.g)
        results = Results(prob.nx, prob.nc)

        return new(
            prob, gpu_prob,
            [], 1, Vector{CuArray{Float64}}(), Vector{SolverIterationLayer}(),
            opts, workspace, results
        )
    end
end