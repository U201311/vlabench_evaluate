from typing import Dict, Any, List, Optional, Sequence, Tuple, Union
import json
import numpy as np
import os 
import sapien
import torch
import random
from transforms3d.euler import euler2quat
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.examples.real2sim_3d_assets.constants import ASSET_3D_PATH, REAL2SIM_3D_ASSETS_PATH, CONTAINER_3D_PATH
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion

OBJ_NAME_LIST = ["pear_1","pear_2", "pear_3",  "lemon_0", "lemon_1", "lemon_2", "lemon_3","lime_0","lime_1","lime_2","banana_1","banana_2"]
ASSET_EVALUATE_JSON_PATH = "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/mani_skill/examples/real2sim_3d_assets/vla_evaluate_task.json"
ASSET_BASE_PATH = "/mnt/data/liy/projects/VLABench/VLABench/assets/obj/meshes"



def generate_random_pos(xy_center, z, half_edge_x, half_edge_y):
    """
    在矩形区域内生成随机坐标点
    参数:
        xy_center (np.array): 中心点的x、y坐标 [x_center, y_center]
        z (float): 固定的z坐标值
        half_edge_x (float): x轴半边长（范围的一半）
        half_edge_y (float): y轴半边长（范围的一半）
    返回:
        np.array: 随机生成的 [x, y, z] 坐标
    """
    # 生成x坐标：中心点 ± 半长范围内的随机值
    x = np.random.uniform(xy_center[0] - half_edge_x, xy_center[0] + half_edge_x)
    # 生成y坐标：中心点 ± 半长范围内的随机值
    y = np.random.uniform(xy_center[1] - half_edge_y, xy_center[1] + half_edge_y)
    # 返回 [x, y, z]
    return np.array([x, y, z])    


def get_objs_random_pose(xy_center, half_edge_length_x, half_edge_length_y, z_value, extents_x, extents_y, quats, threshold_scale=1.0):

    # quat = randomization.random_quaternions(b, lock_x=True, lock_y=True)
    xy_center = np.array(xy_center)
    half_edge_length_x = np.array(half_edge_length_x)
    half_edge_length_y = np.array(half_edge_length_y)
    z_value = np.array(z_value)
    extents_x = np.array(extents_x)
    extents_y = np.array(extents_y)

    if np.linalg.norm([half_edge_length_x[0], half_edge_length_y[0]]) >= np.linalg.norm([half_edge_length_x[1], half_edge_length_y[1]]):
        first_idx, second_idx = 1, 0  
    else:
        first_idx, second_idx = 0, 1

    half_extents_x = extents_x / 2.0
    half_extents_y = extents_y / 2.0

    # threshold = threshold_scale * (max(half_extents_x) + max(half_extents_y)) # 规则长方体/正方体
    threshold = threshold_scale * np.linalg.norm([2 * max(half_extents_x), 2 * max(half_extents_y)]) # 物体的形状不规则
    # threshold = 0.8 * np.mean([half_extents_x[0] + half_extents_x[1], half_extents_y[0] + half_extents_y[1]]) # 物体大小相差比较大
    first_obj_xyz = generate_random_pos(xy_center[first_idx], z_value[first_idx], half_edge_length_x[first_idx], half_edge_length_y[first_idx])

    max_attempts = 1000
    max_distance = -np.inf
    best_obj_xyz = None 
    for _ in range(max_attempts):
        candidate_obj_xyz = generate_random_pos(
            xy_center[second_idx], z_value[second_idx],
            half_edge_length_x[second_idx], half_edge_length_y[second_idx]
        )
        distance = np.linalg.norm(candidate_obj_xyz[:2] - first_obj_xyz[:2])
        
        if distance > max_distance:
            max_distance = distance
            best_obj_xyz = candidate_obj_xyz
        if distance >= threshold:
            break

    second_obj_xyz = best_obj_xyz if best_obj_xyz is not None else candidate_obj_xyz
    if first_idx == 0:
        source_obj_xyz = first_obj_xyz
        target_obj_xyz = second_obj_xyz
    else:
        source_obj_xyz = second_obj_xyz
        target_obj_xyz = first_obj_xyz

    quats = np.array([np.array(quat) if quat is not None else euler2quat(0, 0, np.random.uniform(-np.pi, np.pi),"sxyz") for quat in quats])

    source_obj_quat = quats[0]
    target_obj_quat= quats[1]

    return torch.from_numpy(source_obj_xyz), torch.from_numpy(source_obj_quat), torch.from_numpy(target_obj_xyz), torch.from_numpy(target_obj_quat)  
   
   
def parse_asset_json_file(source_idx):
    """
    Parse the asset json file and return the mjcf_path, texture_path, obj_path, scale, obj_name
    """
    with open(ASSET_EVALUATE_JSON_PATH, "r", encoding="utf-8") as f:    
        data = json.load(f)
    # get the mjcf_path, texture_path, obj_path, scale, obj_name
    obj_name = OBJ_NAME_LIST[source_idx]
    obj_class = obj_name.split("_")[0]
    obj_info = data["fruit"][obj_class][obj_name]
    texture_path = obj_info["material"]
    scale = obj_info["scale"]
    obj_path = obj_info["objfile_name"]
    texture_path = os.path.join(ASSET_BASE_PATH, "fruit", obj_class,obj_name, "visual",obj_info["material"])
    mjcf_path = os.path.join(ASSET_BASE_PATH, "fruit", obj_class, obj_name, obj_info["mjcf_file_name"])
    obj_path = os.path.join(ASSET_BASE_PATH, "fruit",obj_class,obj_name, "visual",obj_info["objfile_name"])
    return mjcf_path, texture_path, obj_path, scale, obj_name


@register_env("AssetsPick-v3", max_episode_steps=500)
class AssetsPickEnvBench(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    
    agent: PandaWristCam
    
    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        self.object_name = kwargs.pop("object_name", "banana_1")
        self.container_name = kwargs.pop("container_name", "bowl")

        self.object = {"name":[],"actor":[]}
        self.consecutive_grasp = 0
        self.reconfiguration_num = 0
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
        # agent was set by self.table_scene.initialize()
        robot_pose = sapien.Pose(p=[-0.615, 0, 0], q=[1,0,0,0])
        # pose = sapien_utils.look_at() # we should not use function look_at()
        eye = torch.tensor([0.76357918+robot_pose.p[0], -0.0395012+robot_pose.p[1], 0.68071344+robot_pose.p[2]])
        rotation = torch.tensor([
                    [-0.53301526, -0.05021884, -0.844614,],
                    [0.01688062, -0.99866954, 0.04872569,],
                    [-0.84593722, 0.01171393, 0.53315383,],
                ])
        pose = Pose.create_from_pq(p=eye, q=matrix_to_quaternion(rotation))
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=640,
                height=480,
                fov=np.deg2rad(44), # vertical fov for realsense d435
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
  
    def get_language_instruction(self, is_place = True):
        object_name = self.object["name"][0].replace("_", " ")
        if is_place:
            container_name = self.object["name"][1].replace("_", " ")
            return [f"Put {object_name} on {container_name}."] * self.num_envs
        return [f"Pick {object_name} up."] * self.num_envs

    def get_scene_description(self):
        return [f"The scene is a simulated workspace designed for robotics tasks. It centers around a wooden table surface situated within a plain, neutral-colored room. A robotic arm is positioned above the table, ready to interact with the environment."] * self.num_envs

    # not useful, for the pose being changed in _initial_episode
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))
    
    
    def _builder_object_helper(self, obj_path_root_path: str,obj_path: str,quat, scale):
        assests_scale_data_path = REAL2SIM_3D_ASSETS_PATH
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
        xyz, quat = self.get_random_pose()
        initial_pose = sapien.Pose(p=xyz, q=quat)
        builder.set_initial_pose(initial_pose)
        return builder.build_dynamic(name=object_name)

        
    def _build_container(self):
        container_root_path = CONTAINER_3D_PATH
        container_path_list = os.listdir(container_root_path)
        container_path_list = list(filter(lambda x: not x.endswith('.ply'), container_path_list)) # pop the ply file path
        q_x_90 = euler2quat(np.pi / 2, 0, 0).astype(np.float32)
        name = self.object["name"].append(container_path_list[0].split('/')[0])
        container_actor = self._builder_object_helper(container_root_path, container_path_list[0], q_x_90, 1)
        return name, container_actor



    def _build_obj_from_mjcf(self, mjcf_path: str, texture_path: str, obj_path: str, scale: float, obj_name: str):
        loader = self.scene.create_mjcf_loader()
        builders = loader.parse(str(mjcf_path))
        actor_builders = builders["actor_builders"]
        print(len(actor_builders))
        builder = actor_builders[0]

        # build material
        # 判断texture_path是否存在
        if not os.path.exists(texture_path):
            print(f"Texture path {texture_path} does not exist.")
            return None
        
        mat = sapien.render.RenderMaterial()
        mat.set_base_color_texture(sapien.render.RenderTexture2D(texture_path))
        
        builder.set_mass_and_inertia(
            mass=0.1,
            cmass_local_pose=sapien.Pose([0,0,0]),
            inertia=[0,0,0], 
        )
        
        builder.add_visual_from_file(
            filename=obj_path,
            scale=scale,
            material=mat,
        )
        builder.add_multiple_convex_collisions_from_file(
            filename=obj_path,
            density=100,
            scale=scale,
            decomposition="coacd"
        )
        
        xyz, quat = self.get_random_pose()
        initial_pose = sapien.Pose(p=xyz, q=quat)
        builder.set_initial_pose(initial_pose)
        return builder.build_dynamic(name=obj_name)


    def get_true_random_pose(self,):
        source_extents = self.object["actor"][0].get_first_collision_mesh(to_world_frame=False).bounding_box_oriented.extents
        target_extents = self.object["actor"][1].get_first_collision_mesh(to_world_frame=False).bounding_box_oriented.extents
        extents_x = [source_extents[0], target_extents[0]]
        extents_y = [source_extents[1], target_extents[1]]

        xy_center = np.array([[-0.3, 0.0],[-0.3, 0.0]])
        half_edge_length_x = np.array([0.1, 0.1])
        half_edge_length_y = np.array([0.2, 0.2])
        z_value = np.array([0,0])
        quats = [None, np.array([1.0, 0.0, 0.0, 0.0])]

        source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat = get_objs_random_pose(
                xy_center, half_edge_length_x, half_edge_length_y,
                z_value, extents_x, extents_y, quats, 0.9,
            )
        return source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat



    def get_random_pose(self, episode_idx=None,):
        xy_center = np.array([-0.3, 0.0])
        step_size = 0.05
        edge_length_x = 0.1*2 # self.table_scene.table_length / 2.0
        edge_length_y = 0.15*2 # self.table_scene.table_width / 2.0

        x_range = np.array([-edge_length_x/2.0, edge_length_x/2.0])
        y_range = np.array([-edge_length_y/2.0, edge_length_y/2.0])

        x_vals = np.arange(x_range[0]+xy_center[0], x_range[1]+xy_center[0] + 0.0001, step_size)
        y_vals = np.arange(y_range[0]+xy_center[1], y_range[1]+xy_center[1] + 0.0001, step_size)

        # x_vals = np.random.uniform(x_range[0]+xy_center[0], x_range[1]+xy_center[0], 110)
        # y_vals = np.random.uniform(y_range[0]+xy_center[1], y_range[1]+xy_center[1], 110)

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
    
    
    def _settle(self, t: int = 0.5):
        """run the simulation for some steps to help settle the objects"""
        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()
    
    
    def _load_scene(self, options:dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()
        self.object = {"name": [],"actor": [],}

        # load the contaienr
        container_path_root_path = CONTAINER_3D_PATH
        container_path_list = os.listdir(container_path_root_path)
        container_path_list = list(filter(lambda x: not x.endswith('.ply'), container_path_list)) # pop the ply file path
        q_x_90 = euler2quat(np.pi / 2, 0, 0).astype(np.float32)

        # load the objects
        source_obj_name = self.object_name
        container_name = self.container_name

        source_idx = [i for i, v in enumerate(OBJ_NAME_LIST) if v == source_obj_name][0]
        container_idx = [i for i, v in enumerate(container_path_list) if v == container_name+".glb"][0]

        
        mjcf_path, texture_path, obj_path, scale, obj_name = parse_asset_json_file(source_idx)
        obj_actor = self._build_obj_from_mjcf(mjcf_path, texture_path, obj_path, scale, obj_name)
        self.object["actor"].append(obj_actor)
        self.object["name"].append(obj_name)

        
        self.object["name"].append(container_path_list[container_idx].split('.')[0])
        actor = self._builder_object_helper(container_path_root_path, container_path_list[container_idx], q_x_90, 1)
        self.object["actor"].append(actor)


    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.set_initial_qpos(env_idx)

            # but not can be used for multi-env
            source_obj_xyz, source_obj_quat, target_obj_xyz, target_obj_quat = self.get_true_random_pose()
            self.object["actor"][0].set_pose(Pose.create_from_pq(p=source_obj_xyz, q=source_obj_quat))
            self.object["actor"][1].set_pose(Pose.create_from_pq(p=target_obj_xyz, q=target_obj_quat))


            # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
            """source object bbox size (3, )"""
            self.episode_source_obj_bbox_world = torch.from_numpy(self.object["actor"][0].get_first_collision_mesh(to_world_frame=True).bounding_box_oriented.extents)
            """target object bbox size (3, )"""
            self.episode_target_obj_bbox_world = torch.from_numpy(self.object["actor"][1].get_first_collision_mesh(to_world_frame=True).bounding_box_oriented.extents)
            
            # stats to track
            self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32)
            self.episode_stats = dict(
                # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
                # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
                is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool),
                # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
                consecutive_grasp=torch.zeros((b,), dtype=torch.bool),
            )
    
    
    def set_initial_qpos(self, env_idx):
        robot_init_qpos_noise = 0
        b = len(env_idx)
        # fmt: off
        qpos = np.array(
            [-0.1788885, -0.5299233, 0.21601543, -2.9509537, 0.16559684, 2.4244094, 0.6683393, 0.04, 0.04],
        )
        # fmt: on
        if self._enhanced_determinism:
            qpos = (
                self._batched_episode_rng[env_idx].normal(
                    0, robot_init_qpos_noise, len(qpos)
                )
                + qpos
            )
        else:
            qpos = (
                self._episode_rng.normal(
                    0, robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
    

    def _get_obs_extra(self, info: Dict):
        """Get task-relevant extra observations. Usually defined on a task by task basis"""
        return dict(
            is_src_obj_grasped=info["is_src_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

    def evaluate(self):
        """
        Evaluate whether the environment is currently in a success state by returning a dictionary with a "success" key or
        a failure state via a "fail" key
        """
        info = self._evaluate(
            success_require_src_completely_on_target=True,
        )
        return info
    
    def _evaluate(
        self,
        success_require_src_completely_on_target=True,
        z_flag_required_offset=0.02,
        **kwargs,
    ):
        source_object: Actor = self.object["actor"][0]
        target_object: Actor = self.object["actor"][1]
        source_obj_pose = source_object.pose
        target_obj_pose = target_object.pose

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.is_grasping(source_object)
        # if is_src_obj_grasped:
        self.consecutive_grasp += is_src_obj_grasped
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
            torch.linalg.norm(offset[:, :2], dim=1)
            <= torch.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
        )
        z_flag = (offset[:, 2] > 0) & (
            offset[:, 2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag & z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contact_forces = self.scene.get_pairwise_contact_forces(
                source_object, target_object
            )
            net_forces = torch.linalg.norm(contact_forces, dim=1)
            src_on_target = src_on_target & (net_forces > 0.05)

        success = src_on_target

        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] | consecutive_grasp
        )

        return dict(**self.episode_stats, success=success)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.object["actor"][0].pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_src_obj_grasped = info["is_src_obj_grasped"]
        reward += is_src_obj_grasped

        return reward

