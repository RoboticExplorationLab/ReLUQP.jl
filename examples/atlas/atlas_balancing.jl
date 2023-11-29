using Pkg; Pkg.activate(joinpath(@__DIR__, ".."));
using ReLUQP
using JLD2
using SparseArrays

include(joinpath(@__DIR__, "../mpc_utils.jl"))
include(joinpath(@__DIR__, "atlas_utils.jl"))

# Setup model and visualizer
atlas = Atlas();
vis = Visualizer();
mvis = init_visualizer(atlas, vis)

# Load balanced reference
@load joinpath(@__DIR__, "atlas_ref.jld2") x_ref u_ref;
visualize!(atlas, mvis, x_ref)

# Calculate discrete dynamics for a balanced position
h = 0.01;
Ad = FD.jacobian(x->rk4(atlas, x, u_ref, h), x_ref);
Bd = FD.jacobian(u->rk4(atlas, x_ref, u, h), u_ref);

# Set up cost matrices (hand-tuned)
Q = spdiagm([1e3*ones(12); repeat([1e1; 1e1; 1e3], 3); 1e1*ones(8); 1e2*ones(12); repeat([1; 1; 1e2], 3); 1*ones(8)]);
R = spdiagm(1e-3*ones(atlas.nu));

# Calculate infinite-horizon LQR cost-to-go and gain matrices
K, Qf = ihlqr(Ad, Bd, Q, R, Q, max_iters = 1000);

# Define additional constraints for the QP (just torques for Atlas)
horizon = 2;
A_torque = kron(I(horizon), [I(atlas.nu) zeros(atlas.nu, atlas.nx)]);
l_torque = repeat(-atlas.torque_limits - u_ref, horizon);
u_torque = repeat(atlas.torque_limits - u_ref, horizon);

# Setup QP
H, g, A, l, u, g_x0, lu_x0 = gen_condensed_mpc_qp(Ad, Bd, Q, R, Qf, horizon, A_torque, l_torque, u_torque, K);

# Setup solver
m = ReLUQP.setup(H, g, A, l, u, verbose = false, eps_primal=1e-2, eps_dual=1e-2, max_iters=10, iters_btw_checks=1);

# Simulate
N = 300;
X = [zeros(atlas.nx) for _ = 1:N];
U = [zeros(atlas.nu) for _ = 1:N];
X[1] = deepcopy(x_ref);
X[1][atlas.nq + 5] = 1.3; # Perturb i.c.

# Warmstart solver
Δx = X[1] - x_ref;
ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx);
m.opts.max_iters = 4000;
m.opts.check_convergence = false;
ReLUQP.solve(m);
m.opts.max_iters = 10;

# Run simulation
for k = 1:N - 1
    # Get error
    global Δx = X[k] - x_ref

    # Update solver
    ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx)

    # Solve and get controls
    results = ReLUQP.solve(m)
    global U[k] = results.x[1:atlas.nu] - K*Δx

    # Integrate
    global X[k + 1] = rk4(atlas, X[k], clamp.(u_ref + U[k], -atlas.torque_limits, atlas.torque_limits), h)
end
animate!(atlas, mvis, X, Δt=h);
readline()