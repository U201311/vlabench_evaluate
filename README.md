# vlabench_evaluate


### step 1: 
* install maniskill environment
  * [ManiSkill GitHub Repository](https://github.com/haosulab/ManiSkill)

### step 2:
* modify :
    * /mani_skill/examples/real2sim_3d_assets/constants.py 中的 REAL2SIM_3D_ASSETS_PATH, CONTAINER_3D_PATH
    * CONTAINER_3D_PATH : plate, bowl 的文件夹路径，格式为.glb格式
    * REAL2SIM_3D_ASSETS_PATH： assets_scale.json 的文件路径（不需要可以删掉）
    * /mani_skill/examples/real2sim_3d_assets/tabletop_pick_and_place_v3 中的常量
    * ASSET_BASE_PATH: /mesh 文件夹的路径
    * ASSET_EVALUATE_JSON_PATH:vla_evaluate_task.json 对应的路径
* bash run.sh