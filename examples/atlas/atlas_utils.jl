# Setup the dynamics and visualization for the Atlas model
using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using Random
using StaticArrays
using Rotations
using LinearAlgebra
using ForwardDiff
import ForwardDiff as FD

const URDFPATH = joinpath(@__DIR__, "urdf", "atlas_all.urdf")

struct Atlas
    mech::Mechanism{Float64}
    statecache::StateCache
    dynrescache::DynamicsResultCache
    nq::Int
    nv::Int
    nx::Int
    nu::Int
    joint_names
    torque_limits::Vector{Float64}
    function Atlas()
        # Create the Mechanism (defaults to floating base)
        mech = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=true)

        ### Pin the left foot to the ground
        foot_frame = findbody(mech, "l_foot")
        pelvis_frame = findbody(mech, "pelvis")
        world_frame = findbody(mech, "world")

        state = MechanismState(mech)
        pelvis_to_foot = translation(relative_transform(state, 
                default_frame(pelvis_frame), default_frame(foot_frame)))

        foot_location = @SVector [pelvis_to_foot[1], pelvis_to_foot[2], -0.08]

        # add fixed joint to ground
        foot_joint = Joint("foot_joint", Fixed{Float64}())
        world_to_joint = Transform3D(frame_before(foot_joint), 
                        default_frame(world_frame), 
                        -foot_location)

        attach!(mech, world_frame, foot_frame, foot_joint, joint_pose=world_to_joint)

        remove_joint!(mech, findjoint(mech, "pelvis_to_world"))

        ## Get mechanism details
        nq = num_positions(mech)
        nv = num_velocities(mech)
        nx = nq + nv
        nu = nq

        joint_names = [joint.name for joint in joints(mech)[2:end]] # Exclude foot joint
        limits_dict = Dict( # Torque TODO: lower, upper
            "l_leg_akx" => [360.],
 		    "l_leg_aky" => [740.],
 		    "l_leg_kny" => [890.],
 		    "l_leg_hpy" => [840.],
 		    "l_leg_hpx" => [530.],
 		    "l_leg_hpz" => [275.],
 		    "back_bkz" => [106.],
 		    "r_leg_hpz" => [275.],
 		    "back_bky" => [445.],
 		    "r_leg_hpx" => [530.],
 		    "back_bkx" => [300.],
 		    "r_leg_hpy" => [840.],
 		    "l_arm_shz" => [87.],
 		    "r_arm_shz" => [87.],
 		    "r_leg_kny" => [890.],
 		    "l_arm_shx" => [99.],
 		    "r_arm_shx" => [99.],
 		    "r_leg_aky" => [740.],
 		    "l_arm_ely" => [63.],
 		    "r_arm_ely" => [63.],
 		    "r_leg_akx" => [360.],
 		    "l_arm_elx" => [112.],
 		    "r_arm_elx" => [112.],
 		    "l_arm_uwy" => [25.],
 		    "r_arm_uwy" => [25.],
 		    "l_arm_mwx" => [25.],
 		    "r_arm_mwx" => [25.],
 		    "l_arm_lwy" => [25.],
 		    "r_arm_lwy" => [25.]
        )
        torque_limits = [limits_dict[name][1] for name in joint_names]

        new(mech, StateCache(mech), DynamicsResultCache(mech),
            nq, nv, nx, nu,
            joint_names, torque_limits)
    end
end

function dynamics(model::Atlas, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    state = model.statecache[T]
    dyn_result = model.dynrescache[T]

    # Set the mechanism state
    copyto!(state, x)

    # Perform forward dynamics
    dynamics!(dyn_result, state, u)

    return [dyn_result.q̇; dyn_result.v̇]
end

function rk4(model::Atlas, x, u, h)
    k1 = dynamics(model, x, u)
    k2 = dynamics(model, x + h/2*k1, u)
    k3 = dynamics(model, x + h/2*k2, u)
    k4 = dynamics(model, x + h*k3, u)
    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)
end

function init_visualizer(model::Atlas, vis::Visualizer)
    delete!(vis)
    meshes_path = joinpath(@__DIR__, "urdf")
    mvis = MechanismVisualizer(model.mech, URDFVisuals(URDFPATH, package_path=[meshes_path]), vis)
    return mvis
end

function visualize!(model::Atlas, mvis::MechanismVisualizer, q)
    set_configuration!(mvis, q[1:model.nq])
end

function animate!(model::Atlas, mvis::MechanismVisualizer, qs; Δt=0.001)
    anim = MeshCat.Animation(mvis.visualizer; fps=convert(Int, floor(1.0 / Δt)))
    for (t, q) in enumerate(qs)
        MeshCat.atframe(anim, t) do 
            set_configuration!(mvis, q[1:model.nq])
        end
    end
    MeshCat.setanimation!(mvis, anim)

    return anim
end