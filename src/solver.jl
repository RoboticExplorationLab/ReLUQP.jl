"""
Sets up a solver for the following quadratic program
min 1\2x'Hx + g'x
subject to l <= Ax <= u

Input:
    H - positive semidefinite matrix
    g - vector
    A - matrix
    l, u - vectors such that l <= u for all elements

Output:
    solver - solver struct containing pre-computed data for the solver
"""
function setup(H::Matrix{Float64}, g::Vector{Float64}, A::Matrix{Float64}, l::Vector{Float64}, 
                u::Vector{Float64}; settings...)
    # TODO: Create options struct
    opts = Options(;settings...)

    # Initialize the problem (on CPU)
    prob = QPProb(H, g, A, l, u)

    # Scale problem, move to GPU
    if opts.scaling
        gpu_prob = ruiz_equilibration(prob)
    else
        gpu_prob = GPU_QPProb(H, g, A, l, u)
    end

    # Instantiate solver
    m = Solver(opts, prob, gpu_prob)

    # Generate penalty parameter options to compute the GPU layers for
    generate_penalty_params!(m)

    # Generate GPU layers
    generate_layers!(m)

    # Print solver details
    if m.opts.verbose
        print_solver_details(m)
    end

    return m
end

"""
Solve the problem contained in m::Solver using ADMM
"""
function solve(m::Solver)
    workspace, opts = m.workspace, m.opts
    gpu_soln = workspace.gpu_soln

    # Init variables
    ρ_ind = workspace.prev_soln_ρ_ind
    if !opts.warmstart
        gpu_soln .= 0
        ρ_ind = m.start_ρ_ind
    end
    ρ = m.ρs[ρ_ind]

    # Print header if in verbose mode
    if opts.verbose
        print_header(m)
    end

    # ADMM iterations
    for k = 1:opts.max_iters
        m.layer_ρs[ρ_ind](gpu_soln)

        # Only compute residuals if checking convergence and/or using an adaptive penalty
        if (opts.check_convergence || opts.adaptive_ρ || opts.verbose) && k % opts.iters_btw_checks == 0
            calculate_residuals!(m)
            
            # Update ρ and ρ_ind
            if opts.adaptive_ρ
                ρ, ρ_ind = update_penalty(m, ρ, ρ_ind)
            end

            # Print solver progress
            if opts.verbose
                if k % (opts.iters_btw_checks*10) == 0
                    print_header(m)
                end
                print_log(m, k, ρ)
            end

            # Check convergence
            if opts.check_convergence
                if check_convergence(m)
                    m.results.iters = k
                    m.workspace.prev_soln_ρ_ind = ρ_ind
                    return generate_results(m)
                end
            end
        end
    end
    m.results.iters = opts.max_iters
    m.workspace.prev_soln_ρ_ind = ρ_ind

    return generate_results(m)
end

"""
Defines a solver iteration in terms of GPU operations. Performs the following in-place on the input `y`:
`in ← W*in + b`
Then clamps the indices specified by `z_inds` between `l` and `u`
"""
function (m::SolverIterationLayer)(in)
    in .= m.W*in + m.b
    z = view(in, m.z_inds)
    clamp!(z, m.l, m.u) # z clamping

    return nothing;
end

using GPUArrays: @kernel, @index, get_backend
"""
Elementwise in-place clamp for GPU arrays
"""
function Base.clamp!(A::AnyGPUArray, l::AnyGPUArray, u::AnyGPUArray)
    @kernel function clamp_kernel!(A, l, u)
        I = @index(Global, Cartesian)
        A[I] = clamp(A[I], l[I], u[I])
    end
    clamp_kernel!(get_backend(A))(A, l, u; ndrange = size(A))
    return A
end

"""
Updates the cost gradient and constraint bounds. Doesn't update prob, only gpu_prob and other solver terms
"""
function update!(m::Solver; g = nothing, l = nothing, u = nothing)
    gpu_prob, prob, opts = m.gpu_prob, m.prob, m.opts
    if (g !== nothing)
        # Move to GPU if necessary
        if !(typeof(g) <: CuArray{Float64})
            g = CuArray(g)
        end

        # Update g_norm (unscaled)
        m.workspace.g_norm = CUDA.norm(g)

        # Scale if necessary
        if opts.scaling
            g = gpu_prob.c*gpu_prob.D.*g
        end

        # Update problem and layers
        gpu_prob.g = g
        for i in 1:length(m.ρs)
            m.layer_ρs[i].b = m.bias_mat_ρs[i]*g
        end
    end
    if (l !== nothing)
        # Move to GPU if necessary
        if !(typeof(l) <: CuArray{Float64})
            l = CuArray(l)
        end

        # Scale if necessary
        if opts.scaling
            l = gpu_prob.E.*l
        end

        # Update problem and layers
        gpu_prob.l = l
        for i in 1:length(m.ρs)
            m.layer_ρs[i].l = l
        end
    end
    if (u !== nothing)
        # Move to GPU if necessary
        if !(typeof(u) <: CuArray{Float64})
            u = CuArray(u)
        end
        # Scale if necessary
        if opts.scaling
            u = gpu_prob.E.*u
        end
        # Update problem and layers
        gpu_prob.u = u
        for i in 1:length(m.ρs)
            m.layer_ρs[i].u = u
        end
    end
    
    return nothing
end

"""
Calculate the (possibly scaled) primal and dual residuals and their normalizers if penalty updating
is on
"""
function calculate_residuals!(m::Solver)
    gpu_prob, workspace, opts = m.gpu_prob, m.workspace, m.opts
    x, z, λ = workspace.x, workspace.z, workspace.λ
    H, g, A = gpu_prob.H, gpu_prob.g, gpu_prob.A
    c, invD, invE = gpu_prob.c, gpu_prob.invD, gpu_prob.invE

    # Primal terms
    term1 = A*x
    term2 = z

    # Dual terms
    term3 = H*x
    term4 = A'*λ
    term5 = g

    # Calculate residuals
    workspace.primal_res = CUDA.norm(invE.*(term1 - term2))
    workspace.dual_res = CUDA.norm(1/c*invD.*(term3 + term4 + term5))

    # If penalty updating is on, calculate residual normalizers
    if opts.adaptive_ρ
        workspace.primal_normalizer = max(CUDA.norm(invE.*term1), CUDA.norm(invE.*term2), 1e-4)
        workspace.dual_normalizer = max(CUDA.norm(invD.*term3)/c, CUDA.norm(invD.*term4)/c, workspace.g_norm, 1e-4)
    end

    return nothing
end

"""
Update the penalty parameter and index based on the primal and dual residual 
"""
function update_penalty(m::Solver, ρ::Float64, ρ_ind::Int64)
    workspace, opts = m.workspace, m.opts

    primal = workspace.primal_res/workspace.primal_normalizer
    dual = workspace.dual_res/workspace.dual_normalizer
    scaling_factor = sqrt(primal/dual)

    if abs(scaling_factor) < 1e-10 || isnan(scaling_factor)
        return ρ, ρ_ind
    end
    ρ = clamp(ρ*scaling_factor, opts.ρ_min, opts.ρ_max)

    if ρ > m.ρs[ρ_ind]*opts.adaptive_ρ_spacing
        ρ_ind += 1
    elseif ρ < m.ρs[ρ_ind]/opts.adaptive_ρ_spacing
        ρ_ind -= 1
    end
    ρ_ind = clamp(ρ_ind, 1, length(m.ρs))

    return ρ, ρ_ind
end

"""
Check convergence based on the primal and dual residuals
"""
function check_convergence(m::Solver)
    workspace, opts = m.workspace, m.opts

    return workspace.primal_res < opts.eps_primal && workspace.dual_res < opts.eps_dual
end

"""
Generate penalty parameters, centered around opts.ρ with geometric spacing
specified by opts.adaptive_ρ_spacing within [opts.ρ_min, opts.ρ_max]
Initializes solver.start_ρ_ind and solver.workspace.prev_soln_ρ_ind
"""
function generate_penalty_params!(m::Solver)
    opts = m.opts

    m.ρs = [opts.ρ]
    if opts.adaptive_ρ
        ρ = opts.ρ/opts.adaptive_ρ_spacing
        while ρ >= opts.ρ_min
            push!(m.ρs, ρ)
            ρ /= opts.adaptive_ρ_spacing
        end
        ρ = opts.ρ*opts.adaptive_ρ_spacing
        while ρ <= opts.ρ_max
            push!(m.ρs, ρ)
            ρ *= opts.adaptive_ρ_spacing
        end
        m.ρs = sort(m.ρs)
    end

    m.start_ρ_ind = argmin(abs.(m.ρs .- opts.ρ))
    m.workspace.prev_soln_ρ_ind = m.start_ρ_ind

    return nothing
end

"""
Generate an ADMM update layer for each ρ in solver.ρs
"""
function generate_layers!(m::Solver)
    gpu_prob, opts = m.gpu_prob, m.opts
    nx, nc = gpu_prob.nx, gpu_prob.nc
    H, g, A, l, u = gpu_prob.H, gpu_prob.g, gpu_prob.A, gpu_prob.l, gpu_prob.u
    z_inds = nx .+ (1:nc)

    # Resize vectors
    resize!(m.layer_ρs, length(m.ρs))
    resize!(m.bias_mat_ρs, length(m.ρs))

    # Create an iteration layer for each penalty parameter
    for (ρ_ind, ρ) in enumerate(m.ρs)
        # Create ρ matrix
        ρ_mat = ρ*ones(nc)
        ρ_mat[u .- l .<= opts.eq_tol] .= ρ*1e3 # Increase penalty for eq constraints
        ρ_mat = CuArray(diagm(ρ_mat))

        # Generate schur matrix inverse
        kkt_sys = H + CuArray(opts.σ*I(nx)) + A'*ρ_mat*A
        K = CuArray(inv(Array(kkt_sys)))

        # Generate iteration weight matrix, bias matrix, and bias vector
        eye_nx = CuArray(I(nx))
        eye_nc = CuArray(I(nc))
        t1 = K*(opts.σ*eye_nx - A'*ρ_mat*A) 
        t2 =           2*K*A'*ρ_mat  
        t3 =             -K*A'
        t4=    A*K*(opts.σ*eye_nx - A'*ρ_mat*A) + A  
        t5 =   2*A*K*A'*ρ_mat - eye_nc
        t6 =   -A*K*A' + inv(ρ_mat)
        t7 =    ρ_mat*A           
        t8 =                  -ρ_mat       
        t9=              eye_nc
        W = [
            K*(opts.σ*eye_nx - A'*ρ_mat*A)           2*K*A'*ρ_mat               -K*A'
            A*K*(opts.σ*eye_nx - A'*ρ_mat*A) + A     2*A*K*A'*ρ_mat - eye_nc     -A*K*A' + inv(ρ_mat)
            ρ_mat*A                             -ρ_mat                    eye_nc
        ]
        B = [-K; -A*K; zeros(nc, nx)]
        b = B*g

        # Create and save layer on GPU
        m.layer_ρs[ρ_ind] = SolverIterationLayer(W, b, l, u, z_inds)
        m.bias_mat_ρs[ρ_ind] = B
    end

    return nothing
end

"""
Move the results from gpu_soln to the CPU, unscale if necessary. Returns results
"""
function generate_results(m::Solver)
    gpu_prob, workspace, opts = m.gpu_prob, m.workspace, m.opts
    results = m.results
    x, z, λ = workspace.x, workspace.z, workspace.λ # views into gpu_soln

    J = 0.5*x'*gpu_prob.H*x + gpu_prob.g'*x
    if opts.scaling
        J = 1/gpu_prob.c*J
        x = gpu_prob.D.*x
        λ = 1/gpu_prob.c*gpu_prob.E.*λ
        z = gpu_prob.invE.*z
    end
    results.J = J
    results.x = Array(x)
    results.z = Array(z)
    results.λ = Array(λ)

    return results
end

"""
Print solver details including dimensions, conditioning, settings, and scaling
"""
function print_solver_details(m::Solver)
    prob, gpu_prob, opts = m.prob, m.gpu_prob, m.opts
    neq = sum(sum([prob.u .- prob.u .<= opts.eq_tol]))
    @printf("\t\tReLU-QP Solver\n")
    @printf("problem:  variables n = %d,\tconstraints m = %d\n", dims(prob)...)
    @printf("          equality = %d,   \tinequality = %d\n", neq, prob.nc - neq)
    @printf("          cond(H) = %1.2e, cond(A) = %1.2e (unscaled)\n", cond(prob.H), cond(prob.A))
    @printf("settings: ρ = %1.2e [%1.2e, %1.2e], σ = %1.2e\n", opts.ρ, opts.ρ_min, opts.ρ_max, opts.σ)
    if opts.adaptive_ρ
        @printf("          ρ_update: on \t\tspacing: %d\n", opts.adaptive_ρ)
    else
        @printf("          ρ_update: off\n")
    end    
    @printf("          scaling: %s\n", ifelse(opts.scaling, "on", "off"))
    @printf("          warmstarting: %s\n", ifelse(opts.warmstart, "on", "off"))
    if opts.check_convergence
        @printf("          check convergence: on (every %d iters)\n", opts.iters_btw_checks)
    else
        @printf("          check_convergence: off\n")
    end
    @printf("          max iters: %d\n", opts.max_iters)
    @printf("          eps_primal: %1.2e\n", opts.eps_primal)
    @printf("          eps_dual: %1.2e\n", opts.eps_dual)

    if opts.scaling
        @printf("scaling:  min_D = %2.2e max_D = %2.2e\n", minimum(gpu_prob.D), maximum(gpu_prob.D))
        @printf("          min_E = %2.2e max_E = %2.2e\n", minimum(gpu_prob.E), maximum(gpu_prob.E))
        @printf("          cost = %2.2e\n", gpu_prob.c)
        @printf("          cond(H) = %2.2e, cond(A) = %2.2e (scaled)\n", cond(Array(gpu_prob.H)), cond(Array(gpu_prob.A)))
    end

    return nothing
end

# Print a header for the log
function print_header(m::Solver)
    @printf("iter         J               ΔJ             ρ            |r_p|∞          |r_d|∞     \n")
    @printf("------------------------------------------------------------------------------------\n")

    return nothing
end

# Log the current iterates/cost/ρ
function print_log(m::Solver, k, ρ)
    gpu_prob, workspace = m.gpu_prob, m.workspace
    H, g, x = gpu_prob.H, gpu_prob.g, workspace.x

    # Calculate cost
    x = workspace.x
    J = 1/2*x'*H*x + g'*x

    @printf("%d", k)
    @printf("\t%1.2e", J)
    @printf("\t%1.2e", J - workspace.J)
    @printf("\t%1.2e", minimum(ρ))
    @printf("\t%1.2e", norm(workspace.primal_res, Inf))
    @printf("\t%1.2e", norm(workspace.dual_res, Inf))
    @printf("\n")

    # Update workspace
    workspace.J = J

    return nothing
end



