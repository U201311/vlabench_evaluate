import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from dataclasses import dataclass
from typing import Annotated
import tyro
from typing import Optional, List
from simpler_env import SIMPLER_ROOT_DIR
from mani_skill.examples.motionplanning.panda.solutions.panda_simpler_mp import(
    PandaSolvePutCarrot,
    PandaSolvePutEggplant,
    PandaSolvePutSpoon,
    PandaSolveStackCube,
)

PANDA_SIMPLER_MP_SOLUTIONS = {
    "PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1": PandaSolveStackCube,
    "PandaPutSpoonOnTableClothInScene-v1": PandaSolvePutSpoon,
    "PandaPutEggplantInBasketScene-v1": PandaSolvePutEggplant,
    "PandaPutCarrotOnPlateInScene-v1": PandaSolvePutCarrot,
}

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1"
    """The environment ID of the task you want to simulate
        f"Environment to run motion planning solver on. Available options are {list(PANDA_SIMPLER_MP_SOLUTIONS.keys())}"
    """
    # "PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1": PandaSolveStackCube,
    # "PandaPutSpoonOnTableClothInScene-v1": PandaSolvePutSpoon,
    # "PandaPutEggplantInBasketScene-v1": PandaSolvePutEggplant,
    # "PandaPutCarrotOnPlateInScene-v1": PandaSolvePutCarrot,
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb+segmentation"
    """Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script."""
    
    num_traj: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 5
    """Number of trajectories to generate"""
    
    only_count_success: bool = False
    """If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos"""
    
    reward_mode: Optional[str] = None
    """Reward mode"""
    
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "cpu"
    """Simulation backend: 'auto', 'cpu', or 'gpu'"""
    
    render_mode: str = "rgb_array"
    """Render mode: 'sensors' or 'rgb_array'"""

    vis: bool = False
    """Whether to open a GUI for live visualization, whether or not to open a GUI to visualize the solution live"""
    
    save_video: bool = False
    """Whether to save videos locally"""
    
    traj_name: Optional[str] = None
    """The name of the trajectory .h5 file that will be created"""
    
    shader: str = "default"
    """Shader used for rendering: 'default', 'rt', or 'rt-fast'"""
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    
    record_dir: str = SIMPLER_ROOT_DIR+"/videos/datasets_mp"
    """Directory to save recorded trajectories"""
    
    num_procs: int = 1
    """Number of processes for parallel trajectory replay (only for CPU backend)"""
    """Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment."""

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos", # "pd_joint_pos", "pd_joint_pos_vel"
        render_mode=args.render_mode,
        reward_mode="none" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        sim_config={
            "sim_freq": 100,
            "control_freq": 10,
        },
    )
    if env_id not in PANDA_SIMPLER_MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(PANDA_SIMPLER_MP_SOLUTIONS.keys())}")
    
    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, time.strftime("%Y%m%d_%H%M%S")),
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=10,
        save_on_reset=False,
        recording_camera_name="3rd_view_camera",
        avoid_overwriting_video = True,
    )
    output_h5_path = env._h5_file.filename
    solve = PANDA_SIMPLER_MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0

    while True:
        try:
            res = solve(env, seed=seed, debug=False, vis=True if args.vis else False, use_rrt=True)
        except Exception as e:
            print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
            res = -1
        except KeyboardInterrupt:
            print("you press ctrl+c to break the while loop.")
            break

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        # import pdb;pdb.set_trace()
        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths),
                    max_episode_length=np.max(solution_episode_lengths,initial=0),
                    min_episode_length=np.min(solution_episode_lengths,initial=0),
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break
    env.close()
    return output_h5_path

def main(args):
    if args.sim_backend != "gpu" and args.num_procs > 1 and args.num_procs <= args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)

if __name__ == "__main__":
    start = time.time()
    mp.set_start_method("spawn")
    parsed_args = tyro.cli(Args)
    main(parsed_args)
    print(f"Total time taken: {time.time() - start}")
