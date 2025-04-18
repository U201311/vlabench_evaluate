from typing import Dict

import numpy as np
import os 
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.building import actors

from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.examples.real2sim_3d_assets import ASSET_3D_PATH

@register_env("TableTopFreeDraw-v2", max_episode_steps=1000)
class TableTopFreeDrawEnv(BaseEnv):
    """
    This is a simple environment demonstrating drawing simulation on a table with a robot arm. There are no success/rewards defined, users can use this code as a starting point
    to create their own drawing type tasks.
    """

    MAX_DOTS = 1010
    """
    The total "ink" available to use and draw with before you need to call env.reset. NOTE that on GPU simulation it is not recommended to have a very high value for this as it can slow down rendering
    when too many objects are being rendered in many scenes.
    """
    DOT_THICKNESS = 0.003
    """thickness of the paint drawn on to the canvas"""
    CANVAS_THICKNESS = 0.02
    """How thick the canvas on the table is"""
    BRUSH_RADIUS = 0.01
    """The brushes radius"""
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    """The colors of the brushes. If there is more than one color, each parallel environment will have a randomly sampled color."""

    SUPPORTED_REWARD_MODES = ["none"]

    SUPPORTED_ROBOTS = ["panda"]
    agent: PandaStick

    def __init__(self, *args, robot_uids="panda_stick", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # we set contact_offset to a small value as we are not expecting to make any contacts really apart from the brush hitting the canvas too hard.
        # We set solver iterations very low as this environment is not doing a ton of manipulation (the brush is attached to the robot after all)
        return SimConfig(
            sim_freq=100,
            control_freq=20,
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
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=960,
            fov=1.2,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()
        # 获取table的size
        table_length = self.table_scene.table_length
        table_width = self.table_scene.table_width
        # build a white canvas on the table
        # self.canvas = self.scene.create_actor_builder()
        # self.canvas.add_box_visual(
        #     half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
        #     material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
        # )
        # self.canvas.add_box_collision(
        #     half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
        # )
        # self.canvas.initial_pose = sapien.Pose(p=[-0.1, 0, self.CANVAS_THICKNESS / 2])
        # self.canvas = self.canvas.build_static(name="canvas")

        self.dots = []
        color_choices = torch.randint(0, len(self.BRUSH_COLORS), (self.num_envs,))
        
        # self.bowl = actors.build_glb_obj(
        #     self.scene,
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_generation_result/bowl.glb",
        #     half_size=1,
        #     name="bowl",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0, 0.5, 0], q=[0.707, 0, 0, 0.707]))
        
        # self.honey_jar = actors.build_glb_obj(
        #     self.scene,
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_50_class_generation/honey_jar.glb",
        #     half_size=1,
        #     name="honey_jar",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0, 0.25, 0], q=[0.707, 0, 0, 0.707]))
        
        # self.coffee_cup = actors.build_glb_obj(
        #     self.scene,
        #     glb_path= "/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/3d_assets/3d_50_class_generation/711_coffee.glb",
        #     half_size=1,
        #     name="coffee_cup",
        #     body_type="dynamic",
        #     add_collision=True,
        #     initial_pose=sapien.Pose(p=[0.1, 0, 0], q=[0.707, 0, 0, 0.707]))
            
        obj_path_root_path = ASSET_3D_PATH
        obj_path_list = os.listdir(obj_path_root_path)
        obj_path_list = list(filter(lambda x: not x.endswith('.ply'), obj_path_list)) # pop the ply file path
        actor_list = []
        q_x_90 = euler2quat(np.pi / 2, 0, 0).astype(np.float32)
        max_num = 10
        for i in range(len(obj_path_list)):
            if i==max_num:
                break
            obj_abs_path = os.path.join(obj_path_root_path, obj_path_list[i])   
            builder = self.scene.create_actor_builder()
            builder.set_mass_and_inertia(
                mass=0.1,
                cmass_local_pose=sapien.Pose([0,0,0],q=q_x_90),
                inertia=[0,0,0], 
            )
            builder.add_multiple_convex_collisions_from_file(
                filename=obj_abs_path,
                scale=(0.1,0.1,0.1),
                pose=sapien.Pose(p=[0, 0, 0],q=q_x_90),
                decomposition="coacd"
            )
            builder.add_visual_from_file(
                filename=obj_abs_path,
                scale=(0.1,0.1,0.1),
                pose=sapien.Pose(p=[0, 0, 0],q=q_x_90),
            )
            #随机生成一个位置
            # x = np.random.uniform(-table_length/2, table_length/2)  # 根据需要调整范围
            # y = np.random.uniform(-table_width/2, (table_width-0.1)/2)  # 根据需要调整范围
            x = -table_length/2 + table_length * i / len(obj_path_list)
            y = -table_width/2 + table_width * i / len(obj_path_list)
            initial_pose = sapien.Pose(p=[x, y, 0])
            builder.set_initial_pose(initial_pose)
            actor = builder.build_dynamic(name=obj_path_list[i]) # build_dynamic
            # import pdb;pdb.set_trace()
        #     actor_list.append(actor)
        # self.dots.append(Actor.merge(actor_list))
                
 
        # for i in range(self.MAX_DOTS):
        #     actors_list = []
        #     if len(self.BRUSH_COLORS) > 1:
        #         for env_idx in range(self.num_envs):
        #             builder = self.scene.create_actor_builder()
        #             builder.add_cylinder_visual(
        #                 radius=self.BRUSH_RADIUS,
        #                 half_length=self.DOT_THICKNESS / 2,
        #                 material=sapien.render.RenderMaterial(
        #                     base_color=self.BRUSH_COLORS[color_choices[env_idx]]
        #                 ),
        #             )
        #             builder.set_scene_idxs([env_idx])
        #             builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        #             actor = builder.build_kinematic(name=f"dot_{i}_{env_idx}")
        #             actors_list.append(actor)
        #         self.dots.append(Actor.merge(actors_list))
        #     else:
        #         builder = self.scene.create_actor_builder()
        #         builder.add_cylinder_visual(
        #             radius=self.BRUSH_RADIUS,
        #             half_length=self.DOT_THICKNESS / 2,
        #             material=sapien.render.RenderMaterial(
        #                 base_color=self.BRUSH_COLORS[0]
        #             ),
        #         )
        #         builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        #         actor = builder.build_kinematic(name=f"dot_{i}")
        #         self.dots.append(actor)
        
                

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE (stao): for simplicity this task cannot handle partial resets
        self.draw_step = 0
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            for dot in self.dots:
                # initially spawn dots in the table so they aren't seen
                dot.set_pose(
                    sapien.Pose(
                        p=[0, 0, -self.DOT_THICKNESS], q=euler2quat(0, np.pi / 2, 0)
                    )
                )

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        # This is the actual, GPU parallelized, drawing code.
        # This is not real drawing but seeks to mimic drawing by placing dots on the canvas whenever the robot is close enough to the canvas surface
        # We do not actually check if the robot contacts the table (although that is possible) and instead use a fast method to check.
        # We add a 0.005 meter of leeway to make it easier for the robot to get close to the canvas and start drawing instead of having to be super close to the table.
        robot_touching_table = (
            self.agent.tcp.pose.p[:, 2]
            < self.CANVAS_THICKNESS + self.DOT_THICKNESS + 0.005
        )
        robot_brush_pos = torch.zeros((self.num_envs, 3), device=self.device)
        robot_brush_pos[:, 2] = -self.DOT_THICKNESS
        robot_brush_pos[robot_touching_table, :2] = self.agent.tcp.pose.p[
            robot_touching_table, :2
        ]
        robot_brush_pos[robot_touching_table, 2] = (
            self.DOT_THICKNESS / 2 + self.CANVAS_THICKNESS
        )
        # move the next unused dot to the robot's brush position. All unused dots are initialized inside the table so they aren't visible
        self.dots[self.draw_step].set_pose(
            Pose.create_from_pq(robot_brush_pos, euler2quat(0, np.pi / 2, 0))
        )
        self.draw_step += 1

        # on GPU sim we have to call _gpu_apply_all() to apply the changes we make to object poses.
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
