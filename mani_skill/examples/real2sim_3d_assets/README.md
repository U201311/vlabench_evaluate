## panda motionplanning data collection with 3d_assets


Here is a simple document for franka panda arm data collection via motion palnning.

Currently, we only support skill: pick, and only support one environment `AssetsPick-v1` for data collection.


### Data collection command

For data collection, you can try:
```bash
python -m mani_skill.examples.real2sim_3d_assets.real2sim_collect \
--save_video --save_data --control_mode pd_ee_pose --num_procs 1 --num_traj 4 --each_object_traj_num 2
```


What you should notice is that we need ee_pose for world model, so that we use `control mode` -> `pd_ee_pose`; `num_procs` means the process of cpu to run the main function; `num_traj` means the total trajectory number; `each_object_traj_num` means you would like to collect `each_object_traj_num` trajectories on each object with different pose.


### very important and you should know
1. You should notice that if the motion planning fail or the planning success but task fail, this trajectory will be dropped until you have earned `num_traj`(a number) success trajectory.
2. You should notice that if you use `num_procs` > 1, maybe you can collect the same data from the same environment, same episode_idx which control the object's pose, the same object. You will unhappily have `num_procs` times same trajectory. So if you need more than one process, you should modify the logic.
3. Now Our random pose is `very limitted` and the number of it is very small. So If you collect more than 10 trajectories, you shuold modify the object random pose generator, and if you do not do that, maybe the policy trained on it will have bad performance.`get_random_pose` in `mani_skill/examples/real2sim_3d_assets/assets_pick_env.py` may help you to deal with it. 
4. You can freely to choose what object to be interacted with in the function `_load_scene` which is defined in `mani_skill/examples/real2sim_3d_assets/assets_pick_env.py`. 
5. If you have the object with bad pose or scale, you can add it in the `assts_scale.json`.
6. Now the grasp pose is computed in `real2sim_solustions`, with the function `compute_grasp_info_by_obb`. If you would like to change the grasp point, you can modify it here.

### visualiztaion command

For visualization the enviroment, you can try:
```bash
python -m mani_skill.examples.teleoperation.interactive_panda -e "AssetsPick-v1"
```