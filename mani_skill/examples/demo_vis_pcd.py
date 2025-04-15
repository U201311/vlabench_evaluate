import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
from mani_skill.utils.common import to_numpy
import trimesh.scene

EXPORT_DATA_PATH="/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/graspnetproject/point_cloud/"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    parser.add_argument("--obj-name",type=str, default="test", help="name of the test")
    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    env: BaseEnv = gym.make(
        args.env_id,
        #obs_mode="rgb+depth+segmentation",
        obs_mode="pointcloud",
        reward_mode="none",
        sensor_configs=sensor_configs,
    )

    obs, _ = env.reset(seed=args.seed)
    obj_name = args.obj_name
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("obs", obs)
        xyz = obs["pointcloud"]["xyzw"][0, ..., :3]
        colors = obs["pointcloud"]["rgb"][0]
        segmentation = obs["pointcloud"]["segmentation"][0]
        # print(obs["sensor_param"]["base_camera"])
        # base_camera_intrinsics = to_numpy(obs["sensor_param"]["base_camera"]["intrinsic_cv"])
        # base_camera_rgb = to_numpy(obs["sensor_data"]["base_camera"]["rgb"])
        # base_camera_depth = to_numpy(obs["sensor_data"]["base_camera"]["depth"])
        # xyz, colors = rgbd_to_pcd(base_camera_intrinsics, base_camera_rgb, base_camera_depth)
        
        # print(obs["sensor_param"])
        # torch.tensor to numpy
        xyz = xyz.cpu().numpy()
        colors = colors.cpu().numpy()
        segmentation = segmentation.cpu().numpy()
        unique_values = np.unique(segmentation)
        print("Unique segmentation values:", unique_values)        # # segmentation = segmentation.cpu().numpy()
        pcd = trimesh.points.PointCloud(xyz, colors)
        # save pcd
        #pcd.export("/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/graspnetproject/point_cloud/point_cloud.ply")
        # 获取segmentation信息
        # segmentation = obs["pointcloud"]["segmentation"]
        # segmentation = segmentation.cpu().numpy()
        # print(segmentation.shape)
        target_segmentation_values = [18]

        mask = np.isin(segmentation[:, 0], target_segmentation_values)
        xyz = xyz[mask]
        colors = colors[mask]
        # torch.tensor to numpy
        pcd_2 = trimesh.points.PointCloud(xyz, colors)
        export_path = EXPORT_DATA_PATH + f"point_cloud_{obj_name}.ply"
        pcd_2.export(export_path)
        #pcd_2.export(f"/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/graspnetproject/point_cloud/point_cloud_{obj_name}.ply")
        #np.save('/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/graspnetproject/point_cloud/point_cloud_tomato.npy', xyz)
        #pcd_2.export("/mnt/data/liy/projects/maniskill_project/3d_asset_branch/ManiSkill/graspnetproject/point_cloud/point_cloud_tomato.ply")
        # view from first camera
        for uid, config in env.unwrapped._sensor_configs.items():
            print(f"Camera {uid}: {config}")
            if isinstance(config, CameraConfig):
                cam2world = obs["sensor_param"][uid]["cam2world_gl"][0]
                camera = trimesh.scene.Camera(uid, (640, 480), fov=(87, 44))
            break
        trimesh.Scene([pcd_2], camera=camera, camera_transform=cam2world).show()

        if terminated or truncated:
            break
    env.close()
    
    
def rgbd_to_pcd(camera_intrinsics, rgb, depth):
    """
    Convert RGBD image to point cloud
    :param rgbd: RGBD image
    :return: Point cloud
    """
    # Get camera intrinsics(1,3,3)
    camera_intrinsics = camera_intrinsics.reshape(3, 3)
    # Get camera intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Get depth image
    depth = depth.squeeze()

    # Create meshgrid of pixel coordinates
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Convert pixel coordinates to camera coordinates
    z = depth / 1000.0  # Convert depth from mm to m
    x_camera = (x - cx) * z / fx
    y_camera = (y - cy) * z / fy

    points = np.stack((x_camera, y_camera, z), axis=-1).reshape(-1, 3)
    # Get RGB values
    rgb = rgb.reshape(-1, 3) / 255.0  # Normalize RGB values to [0, 1]
    
    return points, rgb
    # Stack camera coordinates and RGB values to create point cloud
    #pcd = np.dstack((x_camera, y_camera, z))
    #pcd_rgb = np.dstack((rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]))

    #return pcd.reshape(-1, 3), pcd_rgb.reshape(-1, 3)
   
    

if __name__ == "__main__":
    main(parse_args())
