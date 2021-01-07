#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/1/7 16:29
# @Author : lucky3721
# @Version：V 1.0
# @File : start_example.py
# @desc :
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()