import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.panda_put_on_in_scene import (
    PandaStackGreenCubeOnYellowCubeBakedTexInScene,
    PandaPutSpoonOnTableClothInScene,
    PandaPutEggplantInBasketScene,
    PandaPutCarrotOnPlateInScene,
)
from mani_skill.envs.tasks import StackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver, SimplerCollectPandaArmMotionPlanningSolver, get_numpy
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb, get_grasp_info)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.pose import Pose
import torch
# put source object to target object
def pick_and_place(env, seed=None, debug=False, vis=False, use_rrt=False, simpler_collcet=True):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
        "pd_ee_delta_pose",
        "pd_ee_target_delta_pose",
        "pd_ee_pose",
    ], f"Invalid control mode: {env.unwrapped.control_mode}."

    cls = SimplerCollectPandaArmMotionPlanningSolver if simpler_collcet else PandaArmMotionPlanningSolver
    planner = cls(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.45 # our panda finger
    env = env.unwrapped
    source_obb = get_actor_obb(env.objs[env.source_obj_name])
    target_obb = get_actor_obb(env.objs[env.target_obj_name])

    # All the Pose and direction are defined in the world frame,  the conversion to the robot base(root) frame is handled by the motion planner.
    approaching_dir = np.array([0, 0, -1])
    target_closing = get_numpy(env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1], env.device)
    # grasp_info = get_grasp_info(actor = env.objs[env.source_obj_name], obb = source_obb, 
    #                    depth = FINGER_LENGTH, offset=0, approaching=approaching_dir)
    grasp_info = compute_grasp_info_by_obb(
        source_obb,
        approaching=approaching_dir,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching_dir, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True) if not use_rrt else planner.move_to_pose_with_RRTConnect(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    refine_steps = 0
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose
    planner.move_to_pose_with_screw(reach_pose,refine_steps=refine_steps) if not use_rrt else planner.move_to_pose_with_RRTConnect(reach_pose,refine_steps=refine_steps)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose,refine_steps=refine_steps) if not use_rrt else planner.move_to_pose_with_RRTConnect(grasp_pose,refine_steps=refine_steps)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose,refine_steps=refine_steps) if not use_rrt else planner.move_to_pose_with_RRTConnect(lift_pose,refine_steps=refine_steps)

    # -------------------------------------------------------------------------- #
    # place
    # -------------------------------------------------------------------------- #
    if isinstance(env, PandaPutEggplantInBasketScene):
        offset = np.array([0,0,0.05])
    else:
        offset = np.array([0,0,0.03])

    end_pose = sapien.Pose(p=get_numpy(env.objs[env.target_obj_name].pose.p.squeeze(0),env.device)+offset, q=lift_pose.q)
    planner.move_to_pose_with_screw(end_pose,refine_steps=refine_steps) if not use_rrt else planner.move_to_pose_with_RRTConnect(end_pose,refine_steps=refine_steps)
    res = planner.open_gripper(t=10)
    return res, planner

def PandaSolveStackCube(env: PandaStackGreenCubeOnYellowCubeBakedTexInScene, 
                        seed=None, debug=False, vis=False, use_rrt=False):
    return pick_and_place(env, seed, debug, vis, use_rrt)

def PandaSolvePutSpoon(env: PandaPutSpoonOnTableClothInScene, 
                       seed=None, debug=False, vis=False, use_rrt=False):
    return pick_and_place(env, seed, debug, vis, use_rrt)

def PandaSolvePutCarrot(env: PandaPutCarrotOnPlateInScene,
                         seed=None, debug=False, vis=False, use_rrt=False):
    return pick_and_place(env, seed, debug, vis, use_rrt)

def PandaSolvePutEggplant(env: PandaPutEggplantInBasketScene,
                           seed=None, debug=False, vis=False, use_rrt=False, simpler_collcet=True):
    return pick_and_place(env, seed, debug, vis, use_rrt)