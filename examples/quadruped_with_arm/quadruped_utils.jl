using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using LinearAlgebra
using ForwardDiff
using FiniteDiff
using BlockDiagonals

const RBD = RigidBodyDynamics
const FD = ForwardDiff
const URDFPATH = joinpath(@__DIR__, "widowGo1/urdf/widowGo1.urdf")
const MESHPATH = joinpath(@__DIR__, "widowGo1/meshes/")

struct ArmGo1
    mech::Mechanism
    nq::Int
    nv::Int
    nx::Int
    nu::Int
    nu_leg::Int
    nu_arm::Int
    nλ::Int
    config_names::Vector{String}
    vel_names::Vector{String}
    joint_names::Vector{String}
    foot_names::Vector{String}
    leg_joint_names::Vector{String}
    leg_config_inds::Vector{Int}
    leg_vel_inds::Vector{Int}
    arm_joint_names::Vector{String}
    arm_config_inds::Vector{Int}
    arm_vel_inds::Vector{Int}
    torque_limits::Vector{Float64}
    joint_limits::Vector{Float64}
    μ::Float64 # Friction cone
    function ArmGo1(; μ = 1.0)
        mech = parse_urdf(URDFPATH, floating = true, remove_fixed_tree_joints=false)
        state = MechanismState(mech)

        # Dimensions
        nq = num_positions(mech)
        nv = num_velocities(mech)
        nx = nq + nv
        nλ = 12

        # Joint names
        state = RigidBodyDynamics.MechanismState(mech)
        config_names = Vector{String}()
        for k = 1:nq
            push!(config_names, joints(mech)[RBD.configuration_index_to_joint_id(state, k)].name)
        end
        vel_names = Vector{String}()
        for k = 1:nv
            push!(vel_names, joints(mech)[RBD.velocity_index_to_joint_id(state, k)].name)
        end
        joint_names = [joint.name for joint in joints(mech)]
        foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

        # Leg names and inds
        leg_joint_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", 
                           "RL_hip_joint", "FR_thigh_joint", "FL_thigh_joint", 
                           "RR_thigh_joint", "RL_thigh_joint", "FR_calf_joint", 
                           "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        leg_config_inds, leg_vel_inds = [], []
        for name in leg_joint_names
            push!(leg_config_inds, argmax(config_names .== name))
            push!(leg_vel_inds, argmax(vel_names .== name))
        end

        # Arm names and inds
        arm_joint_names = ["widow_waist", "widow_shoulder", "widow_elbow","widow_forearm_roll", 
                            "widow_wrist_angle", "widow_wrist_rotate", "widow_left_finger", "widow_right_finger"]
        arm_config_inds, arm_vel_inds = [], []
        for name in arm_joint_names
            push!(arm_config_inds, argmax(config_names .== name))
            push!(arm_vel_inds, argmax(vel_names .== name))
        end

        # Control dims
        nu_leg = length(leg_vel_inds)
        nu_arm = length(arm_vel_inds)
        nu = nu_leg + nu_arm

        # Torque limits
        torque_limits = [23.7, 23.7, 23.7, 23.7,
                         23.7, 23.7, 23.7, 23.7,
                         23.7, 23.7, 23.7, 23.7,
                         10, 20, 15, 2, 5, 1, 5, 5]

        # Joint limits
        joint_limits = []

        return new(mech, nq, nv, nx, nu, nu_leg, nu_arm, nλ,
            config_names, vel_names, joint_names, foot_names,
            leg_joint_names, leg_config_inds, leg_vel_inds,
            arm_joint_names, arm_config_inds, arm_vel_inds,
            torque_limits, joint_limits, μ)
    end
end

RBD.MechanismState(model::ArmGo1) = MechanismState(model.mech)
RBD.MechanismState(model::ArmGo1, q::Vector, v::Vector) = MechanismState(model.mech, q, v)
RBD.joints(model::ArmGo1) = joints(model.mech)
RBD.bodies(model::ArmGo1) = bodies(model.mech)
RBD.findbody(model::ArmGo1, arg) = findbody(model.mech, arg)
RBD.root_frame(model::ArmGo1) = root_frame(model.mech)

init_visualizer(model::ArmGo1, vis::Visualizer) = MechanismVisualizer(model.mech, 
                    URDFVisuals(URDFPATH, package_path=[MESHPATH]), vis)

function visualize!(model::ArmGo1, mvis::MechanismVisualizer, q)
    set_configuration!(mvis, q[1:model.nq])
end

function animate!(model::ArmGo1, mvis::MechanismVisualizer, qs; Δt=0.001)
    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
    for (t, q) in enumerate(qs)
        MeshCat.atframe(anim, t) do 
            set_configuration!(mvis, q[1:model.nq])
        end
    end
    MeshCat.setanimation!(mvis, anim)

    return anim
end

function B_func(model::ArmGo1)
    B = zeros(model.nv, model.nu)
    for (r, c) in zip(model.leg_vel_inds, 1:model.nu_leg); B[r, c] = 1; end
    for (r, c) in zip(model.arm_vel_inds, model.nu_leg .+ (1:model.nu_arm)); B[r, c] = 1; end
    return B
end

function foot_kinematics(model::ArmGo1, q)
    foot_locs = []
    state = MechanismState{eltype(q)}(model.mech)
    set_configuration!(state, q[1:go1.nq])
    for name in model.foot_names
        fl_foot = findbody(model, name)
        point_in_world = transform(state, Point3D(default_frame(fl_foot), [0.0; 0; 0]), root_frame(model))
        push!(foot_locs, Vector(point_in_world.v))
    end
    return vcat(foot_locs...)
end

function foot_kinematics_jacobian(model::ArmGo1, q::Vector)
    return FD.jacobian(q -> foot_kinematics(model, q), q)*error_jacobian(model, q)
end

function implicit_euler(go1::ArmGo1, x_k, x_next, u, λ, h)
    q_next = x_next[1:go1.nq]
    v_next = x_next[go1.nq + 1:end]
    ω_next = v_next[1:3]

    state = MechanismState(go1, q_next, v_next)

    # Set up dynamics
    M = mass_matrix(state)
    C = dynamics_bias(state)
    B = B_func(go1)
    J = foot_kinematics_jacobian(go1, q_next)

    # Calculate q̇
    q̇ = [0.5*G(q_next[1:4])*v_next[1:3];        # Rotate body velocity into world
         quat_to_rot(q_next[1:4])*v_next[4:6];  # Convert angular velocity to quat derivative
         v_next[7:end]]                         # Joint velocities

    # Calculate v̇
    v̇ = M \ (B*u + J'*λ - C)

    ẋ = [q̇; v̇]

    # Create residual
    return [
        2*G(x_k[1:4])'*x_next[1:4] - h*ω_next       # Torso rotation
        x_next[5:end] - (x_k[5:end] + h*ẋ[5:end])   # Torso pos/joints/velocities
    ]
end

function pinned_constraint(go1::ArmGo1, x_next, foot_locs)
    q_next = x_next[1:go1.nq]
    return foot_kinematics(go1, q_next) - foot_locs
end

function pinned_implicit_euler(go1::ArmGo1, x_k, x_next, u, λ, h, foot_locs)
    return [
        implicit_euler(go1, x_k, x_next, u, λ, h);
        pinned_constraint(go1, x_next, foot_locs)
    ]
end

function implicit_euler_derivatives(go1::ArmGo1, x0, u0, λ0, h, foot_locs)
    A_k = FiniteDiff.finite_difference_jacobian(temp -> implicit_euler(go1, temp, x0, u0, λ0, h), x0)
    A_next = FiniteDiff.finite_difference_jacobian(temp -> implicit_euler(go1, x0, temp, u0, λ0, h), x0)
    B_u = FiniteDiff.finite_difference_jacobian(temp -> implicit_euler(go1, x0, x0, temp, λ0, h), u0)
    B_λ = FiniteDiff.finite_difference_jacobian(temp -> implicit_euler(go1, x0, x0, u0, temp, h), λ0)
    
    A_k = A_k*error_jacobian(go1, x0)
    A_next = A_next*error_jacobian(go1, x0)

    A = -A_next \ A_k
    B_u = -A_next \ B_u
    B_λ = -A_next \ B_λ

    C = FiniteDiff.finite_difference_jacobian(temp -> pinned_constraint(go1, temp, foot_locs), x0)*error_jacobian(go1, x0)

    return A, B_u, B_λ, C
end

function newton_implicit_euler(go1::ArmGo1, x_k, u, h, foot_locs; max_iters = 10, tol = 1e-14)
    nx, nλ = length(x_k), go1.nλ

    # Init guess
    x_guess = copy(x_k)
    λ_guess = zeros(nλ)
    y_guess = [x_guess; λ_guess] # Solve for x and λ together

    # Form residual function
    rFunc(y) = pinned_implicit_euler(go1, x_k, y[1:nx], u, y[nx .+ (1:nλ)], h, foot_locs)

    # Evaluate residual, check convergence
    r = rFunc(y_guess)
    if norm(r, Inf) < tol
        return y_guess[1:nx], y_guess[nx .+ (1:nλ)]
    end

    # Newton steps
    for _ = 1:max_iters
        # Eval residual jacobian
        dr_dy = FD.jacobian(rFunc, y_guess)

        # Apply attitude jacobian for body attitude derivative
        dr_dy = [dr_dy[:, 1:4]*0.5*G(y_guess[1:4]) dr_dy[:, 5:end]]

        # Solve for step
        Δy = -dr_dy \ r

        # Apply step
        y_guess = [
            L_mult(y_guess[1:4])*axis_angle_to_quat(Δy[1:3])
            y_guess[5:end] + Δy[4:end];
        ]

        # Evaluate residual, check convergence
        r = rFunc(y_guess)
        if norm(r, Inf) < tol
            return y_guess[1:nx], y_guess[nx .+ (1:nλ)]
        end
    end

    if norm(r, Inf) > tol*1e3
        @error "Newton solve did not converge. Final residual: " norm(r, Inf)
    end
    return y_guess[1:nx], y_guess[nx .+ (1:nλ)]
end

### Quaternion stuff
function error_jacobian(model::ArmGo1, x)
    E = zeros(eltype(x), length(x), length(x) - 1)
    E[1:4, 1:3] = 0.5*G(x[1:4])
    E[5:end, 4:end] = I(length(x) - 4)
    return E
end

function state_error(model::ArmGo1, x, x0)
    return [
        quat_to_axis_angle(L_mult(x0[1:4])'*x[1:4])
        x[5:end] - x0[5:end]
    ]
end

function axis_angle_to_quat(ω; tol = 1e-12)
    norm_ω = norm(ω)
    
    if norm_ω >= tol
        return [cos(norm_ω/2); ω/norm_ω*sin(norm_ω/2)]
    else
        return [1; 0; 0; 0]
    end
end

skew(v) = [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]

function G(q)
    qs = q[1]
    qv = q[2:4]
    return [-qv'; qs*I + skew(qv)]
end

function L_mult(q)
    qs = q[1]
    qv = q[2:4]
    return [qs -qv'; qv qs*I + skew(qv)]
end

# Derived from p̂₊ = qp̂q† -> Hp₊ = L_mult(q)R_mult(q)'Hp
function quat_to_rot(q)
    skew_qv = skew(q[2:4])
    return 1.0I + 2*q[1]*skew_qv + 2*skew_qv^2
end

function quat_to_axis_angle(q; tol = 1e-12)
    qs = q[1]
    qv = q[2:4]
    norm_qv = norm(qv)
    
    if norm_qv >= tol
        θ = 2*atan(norm_qv, qs)
        return θ*qv/norm_qv
    else
        return zeros(3)
    end
end

function foot_constraints(go1::ArmGo1, Cd, u_ref, λ_ref, horizon)
    nx, nu, nλ = 52, 20, 12
    # Normal force constraint
    μ = go1.μ/sqrt(2)*5 # Conservative approx
    A1 = kron(I(horizon), [zeros(5*4, nu) BlockDiagonal([[0 0 1; 1 0 μ; -1 0 μ; 0 1 μ; 0 -1 μ] for _ = 1:4]) zeros(5*4, nx)])
    l1 = repeat(-BlockDiagonal([[0 0 1; 1 0 μ; -1 0 μ; 0 1 μ; 0 -1 μ] for _ = 1:4])*λ_ref, horizon)
    u1 = fill(Inf, length(l1))

    # Pinned foot constraint
    A2 = kron(I(horizon), [zeros(12, nu + nλ) Cd])
    b2 = zeros(size(A2, 1))

    # Torque limits
    A3 = kron(I(horizon), [I(nu) zeros(nu, nλ + nx)])
    l3 = repeat(-go1.torque_limits - u_ref, horizon)
    u3 = repeat(go1.torque_limits - u_ref, horizon)

    return [A1; A2; A3], [l1; b2; l3], [u1; b2; u3]
end

function standing_config(model::ArmGo1, quat, height, foot_locs, ang = 45)
    ang = ang*pi/180
    q = [quat; 0; 0; height; zeros(go1.nq - 7)]
    q[go1.leg_config_inds[5:8]] .= ang
    q[go1.leg_config_inds[9:end]] .= -ang*2
    function residual(leg_angs)
        q[go1.leg_config_inds] = leg_angs
        return foot_kinematics(go1, q) - foot_locs
    end
    for k = 1:20
        r = residual(q[go1.leg_config_inds])
        dr_dx = FiniteDiff.finite_difference_jacobian(residual, q[go1.leg_config_inds])
        q[go1.leg_config_inds] -= 0.5*(dr_dx\r)
        # visualize!(go1, mvis, q)vscode-remote://wsl%2Bubuntu/home/arun/GPU_ADMM/benchmarks/problems/quadruped_with_arm/prob_data.jld2
    end
    @assert norm(foot_kinematics(go1, q) - foot_locs, Inf) < 1e-5
    q[go1.arm_config_inds] = [0; -pi/2; pi/2; 0; 0; 0; 0; 0];
    return q
end;
