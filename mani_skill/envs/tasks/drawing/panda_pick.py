from typing import Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig



class TableTopPick(BaseEnv):
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

    SUPPORTED_ROBOTS: ["panda"]
    agent: Panda
    
    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    @property
    def _default_sim_config(self):
        pass

