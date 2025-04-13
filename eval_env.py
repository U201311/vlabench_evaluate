import gymnasium as gym
import torch
from collections import defaultdict
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
env_id = "AssetsPick-v3"
num_eval_envs = 10
env_kwargs = dict(obs_mode="state") # modify your env_kwargs here
eval_envs = gym.make(env_id, num_envs=num_eval_envs, reconfiguration_freq=1, **env_kwargs)
obs, _ = eval_envs.reset(seed=0)
eval_metrics = defaultdict(list)
for _ in range(1):
    action = eval_envs.action_space.sample() # replace with your policy action
    obs, rew, terminated, truncated, info = eval_envs.step(action)
    evaluation_info = eval_envs.evaluate()
    success = evaluation_info["success"]
    print(success)