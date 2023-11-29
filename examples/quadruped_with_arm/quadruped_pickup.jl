using Pkg; Pkg.activate(joinpath(@__DIR__, ".."));
using ReLUQP
using JLD2
using SparseArrays
using ProgressMeter

include(joinpath(@__DIR__, "../mpc_utils.jl"))
include(joinpath(@__DIR__, "quadruped_utils.jl"))

# Setup model and visualizer
go1 = ArmGo1();
vis = Visualizer();
mvis = init_visualizer(go1, vis)

# Load standing reference
@load joinpath(@__DIR__, "quadruped_ref.jld2") x_ref u_ref λ_ref
foot_locs = foot_kinematics(go1, x_ref)
visualize!(go1, mvis, x_ref)

# Calculate discrete dynamics for a balanced position
h = 0.01
Ad, Bd_u, Bd_λ, Cd = implicit_euler_derivatives(go1, x_ref, u_ref, λ_ref, h, foot_kinematics(go1, x_ref));

# Set up cost matrices (hand-tuned)
Q = [500.0*ones(go1.nv); 1e-1*ones(go1.nv)]
Q[go1.arm_config_inds .- 1] .= 50
Q[go1.arm_vel_inds .+ go1.nv] .= 8
Q = spdiagm(Q)
R = spdiagm(1.0e-2*ones(go1.nu));
F = spdiagm(1.0e-2*ones(go1.nλ));#Won't converge

# Calculate infinite-horizon LQR cost-to-go and gain matrices
K, _, Qf = constrained_ihlqr(Ad, Bd_u, Bd_λ, Cd, Q, R, F, Q, max_iters = 1000, tol=1e-7);

# Define additional constraints for the QP (just torques for Atlas)
horizon = 15
A_feet, l_feet, u_feet = foot_constraints(go1, Cd, u_ref, λ_ref, horizon)

# Setup QP
H, g, A, l, u, g_x0, lu_x0 = gen_condensed_mpc_qp(Ad, [Bd_u Bd_λ], Q, [R zeros(20, 12); zeros(12, 20) F], Qf, horizon, A_feet, l_feet, u_feet, [K; zeros(12, 52)])

# Setup solver
m = ReLUQP.setup(H, g, A, l, u, verbose = false, eps_primal=1e-2, eps_dual=1e-2, max_iters=20, iters_btw_checks=1);

# Simulate
h = 0.005
tf = 2
N = Int(floor(tf/h + 1))
X = [zeros(go1.nx) for _ = 1:N]
U = [zeros(go1.nu) for _ = 1:N]
X[1] = deepcopy(x_ref)
ang = -40*π/180
quat = [cos(ang/2); sin(ang/2)*[1; 0; 0]...]
X[1] = [standing_config(go1, quat, 0.3, foot_locs); zeros(go1.nv)]
X[1][go1.arm_config_inds[1:5]] = [90*π/180; 70*π/180; -20*π/180; 0; -90*π/180]; 
visualize!(go1, mvis, X[1])

# Warmstart solver
Δx = state_error(go1, X[1], x_ref)
ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx)
m.opts.max_iters = 20000
m.opts.check_convergence = false
ReLUQP.solve(m);
m.opts.max_iters = 20

@showprogress for k = 1:N - 1
    # Get error
    global Δx = state_error(go1, X[k], x_ref)

    # Update solver
    ReLUQP.update!(m, g = g + g_x0*Δx, l = l + lu_x0*Δx, u = u + lu_x0*Δx)

    # Solve and get controls
    results = ReLUQP.solve(m)
    global U[k] = u_ref + results.x[1:go1.nu] - K*Δx

    # Simulate
    global X[k + 1], _ = newton_implicit_euler(go1, X[k], U[k], h, foot_locs)
end
animate!(go1, mvis, X, Δt=h);
readline()