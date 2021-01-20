#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/1/18 17:52
# @Author : lucky3721
# @Version：V 1.0
# @File : lunar_land_example.py
# @desc :
import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

model = DQN('MlpPolicy', 'LunarLander-v2', verbose=1, exploration_final_eps=0.1, target_update_interval=250)
model.learn(total_timesteps=int(1e5))

# Separate env for evaluation
eval_env = gym.make('LunarLander-v2')

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")