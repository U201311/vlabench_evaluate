#!/bin/bash

# 定义对象名称列表
OBJ_NAME_LIST=("pear_1" "pear_2" "pear_3" "lemon_0" "lemon_1" "lemon_2" "lemon_3" "lime_0" "lime_1" "lime_2" "banana_1" "banana_2")

# 遍历列表中的每个对象名称
for object_name in "${OBJ_NAME_LIST[@]}"
do
    echo "正在评估对象: $object_name"
    python -m mani_skill.examples.real2sim_3d_assets.evaluate.real2sim_eval_maniskill3 \
        -e AssetsPick-v3 \
        --control_mode pd_ee_pose \
        --num_procs 1 \
        --num_traj 100 \
        --each_object_traj_num 10 \
        --object_name "$object_name" \
        --container_name bowl
    echo "完成评估: $object_name"
    echo "---------------------------------"
done

echo "所有对象评估完成！"