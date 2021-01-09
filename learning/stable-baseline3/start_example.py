#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/1/7 14:50
# @Author : lucky3721
# @Version：V 1.0
# @File : start_example.py
# @desc : test
import gym

from stable_baselines3 import A2C

env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()