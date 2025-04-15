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
from mani_skill import PACKAGE_DIR
signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c

# run sh:
# python -m mani_skill.examples.real2sim_3d_assets.evaluate.real2sim_eval_maniskill3  -e AssetsPick-v3  --control_mode pd_ee_pose --num_procs 1 --num_traj 100 --each_object_traj_num 10 



@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "AssetsPick-v3"
    """The environment ID of the task you want to simulate
    """

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb+segmentation+depth"
    """Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script."""
    
    num_traj: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 5
    """Total number of trajectories to generate."""

    control_mode: str = "pd_ee_pose"
    """can be one of pd_joint_pos, pd_joint_pos_vel, pd_ee_delta_pose, pd_ee_target_delta_pose"""

    reward_mode: Optional[str] = None
    """Reward mode"""
    
    seed: int = 0
    
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "cpu"
    """Simulation backend: 'auto', 'cpu', or 'gpu'"""
    
    render_mode: str = "rgb_array"
    """Render mode: 'sensors' or 'rgb_array'"""

    vis: bool = False
    """Whether to open a GUI for live visualization, whether or not to open a GUI to visualize the solution live"""
    
    save_video: bool = True
    """Whether to save videos locally"""
    
    save_data: bool = True
    """Whether to save data locally"""

    shader: str = "default"
    """Shader used for rendering: 'default', 'rt', or 'rt-fast'"""
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    
    record_dir: str = PACKAGE_DIR / "videos/datasets_mp"
    """Directory to save recorded trajectories"""
    
    num_procs: int = 1
    """Number of processes for parallel trajectory replay (only for CPU backend)"""
    """Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment."""

    each_object_traj_num: int = 1
    """Trajectory number of each object to be rocorded."""


def _main(args, proc_id=0, num_traj=10):
    env_id = args.env_id
    kwargs = {
        "object_name":"banana_1",
        "container_name":"bowl"
    }
    env = gym.make(
        env_id,
        **kwargs,
        obs_mode = args.obs_mode,
        num_envs = 1,
        control_mode = args.control_mode, # "pd_joint_pos", "pd_joint_pos_vel", "pd_ee_delta_pose" "pd_ee_target_delta_pose"
        render_mode = args.render_mode,
        reward_mode = "none" if args.reward_mode is None else args.reward_mode,
        sensor_configs = dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        # reconfiguration_freq=1, # for debug
        sim_config = {
            "sim_freq": 500,
            "control_freq": 5,
        },
    )  
    timestamp = args.timestamp
    pbar = tqdm(total=num_traj, desc=f"Processing {env_id}")
    obs, _ = env.reset(seed = args.seed + proc_id)
    # env = RecordEpisode(
    #     env,
    #     output_dir=osp.join(args.record_dir, env_id, timestamp, "videos"),
    #     save_trajectory = False,
    #     save_video=args.save_video,
    #     source_type="motionplanning",
    #     source_desc="official motion planning solution from ManiSkill contributors",
    #     video_fps=30,
    #     save_on_reset=False,
    #     avoid_overwriting_video = True,
    #     max_steps_per_video = 1000,
    #     recording_camera_name = "base_camera",
    #     #recording_camera_name = "hand_camera",  # record hand camera
    # )
    eposide_num = 0
    successes = []
    while True:
        # policy = Policy(env, args,obs)
        steps = 150
        action_episode_length =  0
        ## run polict inference result
        for i in range(steps):
            action = env.action_space.sample()
            info = env.step(action)
            action_episode_length += 1
            success = info[2].item()
            if success :
                print(f"Success in {success} steps")
                successes.append(success)
                if success:
                    pbar.update(1)
                    pbar.set_postfix(
                        dict(
                            success_rate = np.mean(successes),
                            avg_episode_length=np.mean(action_episode_length),
                        )
                    )
                    print(f"Success in {i} steps")
                    break
                env.reset()
            else:
                if action_episode_length >= steps:
                    print(f"Failed in {steps} steps")
                    break
        if eposide_num >= num_traj:
            break
        eposide_num += 1
        
    env.close()



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
    parsed_args = tyro.cli(Args)
    main(parsed_args)
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")