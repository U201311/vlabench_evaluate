## PANDA MOTIONPlANNING IN SIMPLER_ENV

```bash
## save .npy file

# simpler put spoon * 
python -m mani_skill.examples.motionplanning.panda.collect_simpler -e PandaPutSpoonOnTableClothInScene-v1 \
--save_video --save_data --control_mode pd_ee_target_delta_pose --num_procs 1 --num_traj 5

# simpler put eggplant
python -m mani_skill.examples.motionplanning.panda.collect_simpler -e PandaPutEggplantInBasketScene-v1 \
--save_video --save_data --control_mode pd_ee_target_delta_pose --num_procs 1 --num_traj 5

# simpler put carrot
python -m mani_skill.examples.motionplanning.panda.collect_simpler -e PandaPutCarrotOnPlateInScene-v1 \
--save_video --save_data --control_mode pd_ee_target_delta_pose --num_procs 1 --num_traj 5

# simpler stack cube
python -m mani_skill.examples.motionplanning.panda.collect_simpler -e PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1 \
--save_video --save_data --control_mode pd_ee_target_delta_pose --num_procs 1 --num_traj 5 

# In local computer
rsync -avzP scp/ wq3:/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/videos/scp

# If you want local visulization, you should add "--vis", but if you add both "--vis" and "--save_video",
# the video saved might have some error patch in the picture.
# Now we not support "--sim_backend gpu", for the error in motion planning, which is caused by _initialize_episode
# function in panda env.

## save .h5 file
# # stack cube
# python -m mani_skill.examples.motionplanning.panda.run_simpler -e PandaStackGreenCubeOnYellowCubeBakedTexInScene-v1 \
# --only_count_success --traj_name "bingwen" --save_video --num_traj 5 --num_procs 1

```