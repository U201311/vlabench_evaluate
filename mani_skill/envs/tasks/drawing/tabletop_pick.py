from typing import Dict, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import os 
import sapien
import torch
import random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.building import actors

from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.examples.real2sim_3d_assets import ASSET_3D_PATH, REAL2SIM_3D_ASSETS_PATH
import json


@register_env("TableTopPick-v1", max_episode_steps=150)
class TableTopPickEnv(BaseEnv):
    """
    This is a simple environment demonstrating pick an object on a table with a robot arm. 
    There are no success/rewards defined, users can using this environment to make panda pick the object.
    """
    SUPPORTED_REWARD_MODES = ["none"]

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: PandaWristCam

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        self.object = {"name": [],"actor": [],}
        self.consecutive_grasp = 0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # we set contact_offset to a small value as we are not expecting to make any contacts really apart from the brush hitting the canvas too hard.
        # We set solver iterations very low as this environment is not doing a ton of manipulation (the brush is attached to the robot after all)
        return SimConfig(
            sim_freq=500,
            control_freq=5,
            scene_config=SceneConfig(
                contact_offset=0.01,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        #pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        pose = sapien_utils.look_at(eye=[ 0.76357918, -0.0395012 , 0.68071344], target=[0, 0, 0], up=[0, 0, 1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=320,
                height=240,
                fov=1.2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self): # what we use to render the scene
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=640, # 1280
            height=480, # 960
            fov=1.2,
            near=0.01,
            far=100,
        )

    def get_language_instruction(self):
        object_name = self.object["name"][0]
        return [f"Pick the {object_name} up."] * self.num_envs

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))


    def get_random_pose(self, episode_idx):
        xy_center = np.array([0.0, 0.0])
        step_size = 0.2
        edge_length_x = 0.2 # self.table_scene.table_length / 2.0
        edge_length_y = 0.6 # self.table_scene.table_width / 2.0

        x_range = np.array([-edge_length_x/2.0, edge_length_x/2.0])
        y_range = np.array([-edge_length_y/2.0, edge_length_y/2.0])

        x_vals = np.arange(x_range[0]+xy_center[0], x_range[1]+xy_center[0] + step_size, step_size)
        y_vals = np.arange(y_range[0]+xy_center[1], y_range[1]+xy_center[1] + step_size, step_size)

        X, Y = np.meshgrid(x_vals, y_vals)
        grid_positions = np.vstack([X.ravel(), Y.ravel()]).T  # transer to shape (N, 2)

        if isinstance(episode_idx, int):
            pos_episode_ids = episode_idx % grid_positions.shape[0]
        else:
            pos_episode_ids = np.random.randint(0, grid_positions.shape[0])

        z = 0.0
        xyz = np.append(grid_positions[pos_episode_ids], z) 
        quat = np.array([1,0,0,0])

        return xyz, quat

    def _builder_object_helper(self, episode_idx, obj_path_root_path: str,obj_path: str,quat, scale = 1.0):
        assests_scale_data_path = REAL2SIM_3D_ASSETS_PATH+"/assets_scale.json"
        with open(assests_scale_data_path, "r", encoding="utf-8") as f:
            assests_scale_data = json.load(f)
        object_name = obj_path.split('.')[0]
        if object_name in assests_scale_data.keys():
            scale = np.array(assests_scale_data[object_name]["scale"])
            quat = np.array(assests_scale_data[object_name]["quat"])
        else:
            scale = [scale] * 3
        obj_abs_path = os.path.join(obj_path_root_path, obj_path)   
        builder = self.scene.create_actor_builder()
        builder.set_mass_and_inertia(
            mass=0.1,
            cmass_local_pose=sapien.Pose([0,0,0],q=quat),
            inertia=[0,0,0], 
        )
        builder.add_multiple_convex_collisions_from_file(
            filename=obj_abs_path,
            scale=scale,
            pose=sapien.Pose(p=[0, 0, 0],q=quat),
            decomposition="coacd"
        )
        builder.add_visual_from_file(
            filename=obj_abs_path,
            scale=scale,
            pose=sapien.Pose(p=[0, 0, 0],q=quat),
        )
        xyz, quat = self.get_random_pose(episode_idx)
        initial_pose = sapien.Pose(p=xyz, q=quat)
        builder.set_initial_pose(initial_pose)
        return builder.build_dynamic(name=object_name)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # with torch.device(self.device):
        self.object = {"name": [],"actor": [],}
        obj_path_root_path = ASSET_3D_PATH
        obj_path_list = os.listdir(obj_path_root_path)
        obj_path_list = list(filter(lambda x: not x.endswith('.ply'), obj_path_list)) # pop the ply file path
        q_x_90 = euler2quat(np.pi / 2, 0, 0).astype(np.float32)
        if "episode_id" in options:
            obj_idx = options["episode_id"] % len(obj_path_list)
            episode_id = options["episode_id"]
        else:
            obj_idx = np.random.randint(0, len(obj_path_list))
            episode_id = None
        # obj_idx = torch.tensor([0])
        self.object["name"].append(obj_path_list[obj_idx].split('.')[0])
        actor = self._builder_object_helper(episode_id, obj_path_root_path, obj_path_list[obj_idx], q_x_90, 1)
        self.object["actor"].append(actor)
        print("created object: ", self.object["name"])
        self.table_scene.initialize(env_idx)

    def _get_obs_extra(self, info: Dict):
        """Get task-relevant extra observations. Usually defined on a task by task basis"""
        return dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
    
    def evaluate(self,) -> dict:
        """
        Evaluate whether the environment is currently in a success state by returning a dictionary with a "success" key or
        a failure state via a "fail" key
        """
        is_grasped = self.agent.is_grasping(self.object["actor"][0])
        self.consecutive_grasp += is_grasped
        self.consecutive_grasp[is_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5
        # self.scene.get_pairwise_contact_forces()
        # maybe can add a z_offset to ensure the pick movement

        return dict(is_grasped = is_grasped, consecutive_grasp = consecutive_grasp)


    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.object["actor"][0].pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        return reward


