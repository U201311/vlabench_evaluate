import multiprocessing as mp
import os
from copy import deepcopy
import time
import signal
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from dataclasses import dataclass
from typing import Annotated
import torch
import tyro
from PIL import Image
from pathlib import Path
from typing import Optional, List
from simpler_env import SIMPLER_ROOT_DIR
from mani_skill.examples.motionplanning.panda.solutions.panda_simpler_mp import(
    PandaSolvePutCarrot,
    PandaSolvePutEggplant,
    PandaSolvePutSpoon,
    PandaSolveStackCube,
)

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
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

    control_mode: str = "pd_joint_pos"
    """can be one of pd_joint_pos, pd_joint_pos_vel, pd_ee_delta_pose, pd_ee_target_delta_pose"""

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
    
    save_data: bool = False
    """Whether to save data locally"""

    shader: str = "default"
    """Shader used for rendering: 'default', 'rt', or 'rt-fast'"""
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    
    record_dir: str = SIMPLER_ROOT_DIR+"/videos/datasets_mp"
    """Directory to save recorded trajectories"""
    
    num_procs: int = 1
    """Number of processes for parallel trajectory replay (only for CPU backend)"""
    """Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment."""

def _main(args, proc_id: int = 0, single_num: int = 0) -> str:
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        num_envs = 1,
        control_mode=args.control_mode, # "pd_joint_pos", "pd_joint_pos_vel", "pd_ee_delta_pose" "pd_ee_target_delta_pose"
        render_mode=args.render_mode,
        reward_mode="none" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        sim_config = {
            "sim_freq": 500,
            "control_freq": 5,
        },
    )
    if env_id not in PANDA_SIMPLER_MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(PANDA_SIMPLER_MP_SOLUTIONS.keys())}")

    timestamp = args.timestamp
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, timestamp, "videos"),
        save_trajectory = False,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=10,
        save_on_reset=False,
        recording_camera_name="3rd_view_camera",
        avoid_overwriting_video = True,
        max_steps_per_video = 1000,
    )
    solve = PANDA_SIMPLER_MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    if single_num==0:
        single_num = args.num_traj
    pbar = tqdm(range(single_num), desc=f"proc_id: {proc_id}")
    seed = proc_id * single_num
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    failure = 0
    error_cnt = 0

    while True:
        try:
            start_solve_t = time.time(); print("start motionplanning!")
            env_reset_options = {"episode_id": torch.tensor([seed]), "reconfigure": False}
            obs, info = env.reset(seed=seed, options=env_reset_options)
            res, planner = solve(env, seed=seed, debug=False, vis=True if args.vis else False, use_rrt=False)
            print("motionplanning using time(s):", time.time()-start_solve_t)
            error_cnt = 0
        except Exception as e:
            print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
            print("<<<<<<<Failure, Motion planning can not get a solution.>>>>>>>")
        else:
            success = res[-1]["continuous_success"].item() # success
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
            traj_data = planner.get_trajectory_data()
            print("*********Successfully, Motion planning get a solution.*******")
        successes.append(success)

        if success:
            # save data
            if args.save_video:
                env.flush_video(name=f"success_proc_{proc_id}_numid_{passed}_epsid_{seed}.mp4")
            if args.save_data:
                exp_dir = Path(args.record_dir) / env_id / timestamp / "data"
                exp_dir.mkdir(parents=True, exist_ok=True)
                res = traj_data.copy()
                res["image"] = [Image.fromarray(im).convert("RGB") for im in res["image"]]
                saving_path = exp_dir / f"success_data_proc_{proc_id}_numid_{passed:0>4d}_epsid_{seed}.npy"
                np.save(saving_path, res)
                print(f"save data at {saving_path}.")
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
            passed += 1
            if passed == single_num:
                break
        else:
            if args.save_video:
                env.flush_video(name = f"failure_proc_{proc_id}_numid_{failure}_epsid_{seed}.mp4", save=True)
            failure += 1
        seed += 1
    env.close()
    return

def main(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.timestamp = timestamp
    if args.sim_backend != "gpu" and args.num_procs > 1 and args.num_procs <= args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        total_num_traj = args.num_traj
        single_num_traj = total_num_traj // args.num_procs
        proc_args = [(deepcopy(args), i, single_num_traj) 
                     for i in range(args.num_procs)]
        pool = mp.Pool(args.num_procs)
        pool.starmap(_main, proc_args)
        pool.close()
        pool.join()
    else:
        _main(args)

if __name__ == "__main__":
    start = time.time()
    mp.set_start_method("spawn")
    parsed_args = tyro.cli(Args)
    main(parsed_args)
    print(f"Total time taken: {time.time() - start}")
