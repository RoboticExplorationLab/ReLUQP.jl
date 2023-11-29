using Printf

function ihlqr(A, B, Q, R, Qf; max_iters = 1000, tol = 1e-8)
    P = Qf
    K = zero(B')
    for _ = 1:max_iters
        P_prev = deepcopy(P)
        K = (R .+ B'*P*B) \ (B'*P*A)
        P = Q + A'P*(A - B*K)
        if norm(P - P_prev, 2) < tol
            return K, P
        end
    end
    @error "ihlqr didn't converge", norm(K - (R .+ B'*P*B) \ (B'*P*A), 2)
    return K, P
end

### Infinite-horizon LQR gain for a state-constrained system, where C is the state constraint
# and B_λ are environmental forces that enforce the constraint. F can be used to add cost to these
# forces if you want to limit how hard the robot pushes against the constraints.
function constrained_ihlqr(A, B_u, B_λ, C, Q, R, F, Qf; max_iters = 1000, tol = 1e-8)
    nu, nλ = size(B_u, 2), size(B_λ, 2)
    P = Qf
    K = zero(B_u')
    L = zero(B_λ)
    for k = 1:max_iters
        P_prev = deepcopy(P)

        # Calculate constrainted L and k
        kkt_lhs = [R + B_u'*P*B_u       B_u'*P*B_λ         B_u'*C';
                       B_λ'*P*B_u   F + B_λ'*P*B_λ         B_λ'*C';
                            C*B_u            C*B_λ   zeros(12, 12)];
        kkt_rhs = [B_u'*P*A; B_λ'*P*A; C*A];

        @assert rank(kkt_lhs) == size(kkt_lhs, 1) @sprintf "%d %d" rank(kkt_lhs) size(kkt_lhs, 1)
        if cond(kkt_lhs) > 1e11
            @warn "KKT is ill-conditioned: " cond(kkt_lhs)
        end

        gains = kkt_lhs \ kkt_rhs;

        K = gains[1:nu, :];
        L = gains[nu .+ (1:nλ), :];
        N = gains[nu + nλ + 1:end, :];
       
        # Update P
        Ā = A - B_u*K - B_λ*L
        # P = Q + K'*R*K + Ā'*P*Ā
        P = Q + A'*P*Ā - A'*C'*N
        if norm(P - P_prev, 2) < tol
            return K, L, P
        elseif k == max_iters
            println(norm(P - P_prev, 2))
        end
    end
    @error "ihlqr didn't converge"
    return K, L, P
end

function gen_sparse_mpc_qp(Ad, Bd, Q, R, Qf, horizon; A_add = Nothing, l_add = Nothing, u_add = Nothing)
    nx, nu = size(Ad, 1), size(Bd, 2)

    # Cost
    H = blockdiag([blockdiag(R, Q) for k = 1:horizon - 1]..., R, sparse(Qf))
    g = zeros(size(H, 1))

    # Dynamics constraint
    A = kron(I(horizon), [Bd -I])
    A[nx + 1:end, nu + 1:end - nx] += kron(I(horizon - 1), [Ad zeros(nx, nu)])
    l = zeros(size(A, 1))
    u = zeros(size(A, 1))

    # If there are additional constraints, append them
    if A_add != Nothing
        A = [A; A_add]
        l = [l; l_add]
        u = [u; u_add]
    end

    return Matrix(H), g, A, l, u
end

function gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_add, l_add, u_add, K = Nothing)
    nx, nu = size(Ad, 1), size(Bd, 2)

    # Gain matrix defaults to zero
    if K === Nothing
        K = zeros(nu, nx)
    end

    # Get the sparse problem (without additional constraints)
    H_sp, g_sp, _ = gen_sparse_mpc_qp(Ad, Bd, Q, R, Qf, horizon)

    # Define F and G such that z = Fz̄ + Gx0 to transform the sparse problem into the dense one
    # z = [u0; x1; u1; x2; u2; ...]
    # z̄ = [Δu0; Δu1; Δu2; ...] where uk = -Kxk + Δuk
    F = kron(Diagonal(I, horizon), [I; Bd])
    for k = 1:horizon - 1
        F += kron(diagm(-k => ones(horizon - k)), 
                [ -K*(Ad - Bd*K)^(k - 1)*Bd;
                    (Ad - Bd*K)^k*Bd        ])
    end

    G = vcat([[-K*(Ad - Bd*K)^(k - 1); (Ad - Bd*K)^k] for k = 1:horizon]...)

    # Convert the sparse problem cost
    H = F'*H_sp*F
    g_x0 = F'*H_sp*G # Initial condition becomes part of cost
    g = g_x0*zeros(nx) + F'*g_sp

    # Add additional constraints if they exist
    A = A_add*F
    lu_x0 = -A_add*G # Initial condition is also part of the constraints

    return H, g, A, l_add, u_add, g_x0, lu_x0
end
        