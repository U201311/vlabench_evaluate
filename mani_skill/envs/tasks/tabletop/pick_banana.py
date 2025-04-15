from typing import Any, Dict, Union,cast

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent
import os 
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from mani_skill.examples.real2sim_3d_assets import ASSET_3D_PATH


@register_env("PickBanana-v1", max_episode_steps=50)
class PickBananaEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict, initial_agent_poses: Optional[Union[sapien.Pose, Pose]] = None):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # self.umbrella = actors.build_glb_obj(
        #     self.scene,
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/banana/7193-纹理.glb",
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/desk/fuzi.glb",
        #     #glb_path=   "/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/desk/bowl.glb",
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/bowl/shoes.glb",
        #     #glb_path = "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/bowl/base.glb",
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_generation_result/umbrella.glb",
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/banana/banana.obj",
        #     half_size=0.2,
        #     name="umbrella",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0, 0, 0], q=[0.707, 0, 0, 0.707]))
        
        # self.bottle = actors.build_glb_obj(
        #     self.scene,
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/banana/7193-纹理.glb",
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/desk/fuzi.glb",
        #     #glb_path=   "/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/desk/bowl.glb",
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/bowl/shoes.glb",
        #     #glb_path = "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/bowl/base.glb",
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_generation_result/bottle.glb",
        #     #glb_path="/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/banana/banana.obj",
        #     half_size=0.1,
        #     name="bottle",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0, 0.5, 0], q=[0.707, 0, 0, 0.707]))
        
        self.bowl = actors.build_glb_obj(
            self.scene,
            glb_path= ASSET_3D_PATH+"bowl.glb",
            half_size=0.1,
            name="bowl",
            body_type="dynamic",
            add_collision=True,
            initial_pose=sapien.Pose(p=[0, 0.5, 0], q=[0.707, 0, 0, 0.707]))
            
        self.pen = actors.build_glb_obj(
            self.scene,
            glb_path= ASSET_3D_PATH+"/pen.glb",
            half_size=1,
            name="pen",
            body_type="dynamic",
            add_collision=True,
            initial_pose=sapien.Pose(p=[0, 0.25, 0], q=[0, 0, 0, 0.707]))
        
        # self.cola = actors.build_glb_obj(
        #     self.scene,
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_generation_result/cola.glb",
        #     half_size=1,
        #     name="cola",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0.25, 0, 0], q=[0.707, 0, 0, 0.707]))
        
        # self.cup = actors.build_glb_obj(
        #     self.scene,
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_generation_result/cup.glb",
        #     half_size=1,
        #     name="cup",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[-0.25, 0, 0], q=[0.707, 0, 0, 0.707]))
        
        # self.paper_cup = actors.build_glb_obj(
        #     self.scene,
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_generation_result/paper_cup.glb",
        #     half_size=1,
        #     name="paper_cup",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0, 0.15, 0], q=[0.707, 0, 0, 0.707]))
        
        # self.strawberry = actors.build_glb_obj(
        #     self.scene,
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_generation_result/strawberry.glb",
        #     half_size=0.5,
        #     name="strawberry",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[-0.15, 0.35, 0], q=[0.707, 0, 0, 0.707]))
       
        self.banana  = actors.build_glb_obj(
            self.scene,
            glb_path= ASSET_3D_PATH+"banana.glb",
            half_size=0.1,
            name="banana",
            body_type="dynamic",
            add_collision=True,
            initial_pose=sapien.Pose(p=[0.127, -0.1, 0], q=[0.382, -0.421, 0.609, 0.553]))
        
         

        # actor: Actor = self.banana  
        # for i, obj in enumerate(actor._objs):
        #     rigid_body_component: PhysxRigidBodyComponent = obj.find_component_by_type(PhysxRigidBodyComponent)
        #     if rigid_body_component is not None:
        #         # note the use of _batched_episode_rng instead of torch.rand. _batched_episode_rng helps ensure reproducibility in parallel environments.
        #         rigid_body_component.mass = self._batched_episode_rng[i].uniform(low=0.1, high=1)
            
        #     # modifying per collision shape properties such as friction values
        #     # for shape in obj.get_collision_shapes():
        #     #     shape.physical_material.dynamic_friction = self._batched_episode_rng[i].uniform(low=0.1, high=0.3)
        #     #     shape.physical_material.static_friction = self._batched_episode_rng[i].uniform(low=0.1, high=0.3)
        #     #     shape.physical_material.restitution = self._batched_episode_rng[i].uniform(low=0.1, high=0.3)
            
        #     render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
        #     for render_shape in render_body_component.render_shapes:
        #         for part in render_shape.parts:
        #             # you can change color, use texture files etc.
        #             part.material.set_base_color(self._batched_episode_rng[i].uniform(low=0., high=1., size=(3, )).tolist() + [1])

        #             # note that textures must use the sapien.render.RenderTexture2D 
        #             # object which allows passing a texture image file path
        #             part.material.set_base_color_texture(None)
        #             part.material.set_normal_texture(None)
        #             part.material.set_emission_texture(None)
        #             part.material.set_transmission_texture(None)
        #             part.material.set_metallic_texture(None)
        #             part.material.set_roughness_texture(None)

            

        # self.pack = actors.build_glb_obj(
        #     self.scene,
        #     glb_path="/mnt/data/liy/projects/maniskill_project/liy_branch/ManiSkill/mani_skill/utils/building/actors/assets/desk/bowl.glb",
        #     half_size=0.01,
        #     name="pack",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0, 0.25, self.cube_half_size], q=[0.7071,0.7071, 0.7071, 0.7071]),
        # )
        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.banana.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.banana.pose.raw_pose,
                tcp_to_obj_pos=self.banana.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.banana.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.banana.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.banana)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.banana.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.banana.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
