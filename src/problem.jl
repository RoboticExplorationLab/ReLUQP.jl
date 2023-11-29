# Define the quadratic programming problem and any helpers
mutable struct QPProb
    H::Matrix{Float64}          # Quadratic cost term
    g::Vector{Float64}          # Linear cost term
    A::Matrix{Float64}          # Constraint Matrix
    l::Vector{Float64}          # Constraint upper bound
    u::Vector{Float64}          # Constraint lower bound
    nx::Int64                   # Number of variables
    nc::Int64                   # Number of constraints
    function QPProb(H, g, A, l, u)
        return new(H, g, A, l, u, size(H, 1), size(A, 1))
    end
end

dims(prob::QPProb) = prob.nx, prob.nc
cost(prob::QPProb, x) = 0.5*x'*prob.H*x + x'*prob.g
constraint(prob::QPProb, x) = prob.A*x
primal_residual(prob::QPProb, x, z) = prob.A*x - z
dual_residual(prob::QPProb, x, λ) = prob.H*x + prob.g + prob.A'*λ

mutable struct GPU_QPProb
    H::CuArray{Float64, 2}      # Quadratic cost term
    g::CuArray{Float64, 1}      # Linear cost term
    A::CuArray{Float64, 2}      # Constraint Matrix
    l::CuArray{Float64, 1}      # Constraint upper bound
    u::CuArray{Float64, 1}      # Constraint lower bound
    nx::Int64                   # Number of variables
    nc::Int64                   # Number of constraints
    D::CuArray{Float64, 1}      # Cost scaling matrix diagonal
    invD::CuArray{Float64, 1}   # Cost scaling matrix diagonal (inverse)
    E::CuArray{Float64, 1}      # Constraint scaling matrix diagonal
    invE::CuArray{Float64, 1}   # Constraint scaling matrix diagonal (inverse)
    c::Float64                  # Cost scaling gain
    function GPU_QPProb(H, g, A, l, u; D = ones(size(H, 1)), E = ones(size(A, 1)), c = 1.0)
        nx, nc = size(H, 1), size(A, 1)
        
        return new(
            CuArray(H), CuArray(g), CuArray(A), CuArray(l), CuArray(u), 
            nx, nc,
            CuArray(D), CuArray(1 ./D), CuArray(E), CuArray(1 ./E), c
        )
    end
end

"""
Scales the provided QPProb on the CPU and returns the scaled problem
on the GPU
"""
function ruiz_equilibration(prob::QPProb)
    H, g, A, l, u = prob.H, prob.g, prob.A, prob.l, prob.u
    n, m = dims(prob)

    H̄ = copy(H)
    ḡ = copy(g)
    Ā = copy(A)
    l̄ = copy(l)
    ū = copy(u)

    D = I(n)
    E = I(m)
    c = 1.0;

    # Helper functions
    colnorm(x, p) = mapslices(temp -> norm(temp, p), x, dims = 1)
    rownorm(x, p) = mapslices(temp -> norm(temp, p), x, dims = 2)

    for i = 1:10
        ### Matrix equilibration
        # Compute norms of first and second part of KKT system
        D_temp = max.(colnorm(H̄, Inf), colnorm(Ā, Inf))'[:]
        E_temp = rownorm(Ā, Inf)[:]

        # Clamp scaling values
        D_temp = min.(max.(D_temp, 1e-4), 1e+4)
        E_temp = min.(max.(E_temp, 1e-4), 1e+4)

        # Take square roots, reciprocal, turn into diag matrices
        D_temp = Diagonal(1 ./sqrt.(D_temp))
        E_temp = Diagonal(1 ./sqrt.(E_temp))

        # Update problem data
        H̄ = D_temp*H̄*D_temp
        ḡ = D_temp*ḡ
        Ā = E_temp*Ā*D_temp

        # Update eq matrices
        D = D*D_temp
        E = E*E_temp

        ### Cost normalization
        D_temp = colnorm(H̄, Inf)
        c_temp = norm(D_temp, 1) / n

        g_inf_norm = min(max(norm(ḡ, Inf), 1e-4), 1e+4)

        c_temp = min(max(max(c_temp, g_inf_norm), 1e-4), 1e+4)

        c_temp = 1/c_temp

        # Update problem data
        H̄ = c_temp.*H̄
        ḡ = c_temp.*ḡ

        # Update cost scaling
        c = c*c_temp
    end

    l̄ = E*l̄
    ū = E*ū

    return GPU_QPProb(H̄, ḡ, Ā, l̄, ū; D=diag(D), E=diag(E), c=c)
end
