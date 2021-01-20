#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/1/20 17:24
# @Author : lucky3721
# @Version：V 1.0
# @File : astk_q_learning.py
# @desc :
import numpy as np
import pandas as pd
import datetime as dt


times = [935, 940, 945, 950, 955, 1000, 1005, 1010, 1015,
         1020, 1025, 1030, 1035, 1040, 1045, 1050, 1055,
         1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305,
         1310, 1315, 1320, 1325, 1330, 1335, 1340, 1345,
         1350, 1355, 1400, 1405, 1410, 1415, 1420, 1425,
         1430, 1435, 1440, 1445, 1450, 1455, 1500]


class Market:
    def __init__(self, data):
        self.action_space = ['B', 'S', 'W']  # 买进、卖出、观望
        self.n_actions = len(self.action_space)
        self.data = data  # 935 940 ... 1500 48根K线的数据
        self.time = 935
        pass

    def step(self, action):
        # 要知道当前在那个状态即时间点,用下一时间点的R（收益）作为
        # 当前采取action的reward
        tix = times.index(self.time)
        nix = tix + 1
        if self.time == 1500:
            reward = 0
            done = True
            s_ = 'terminal'
            # print('time is over.')
        else:
            reward = self.data.R.iloc[nix]
            done = False
            s_ = times[nix]
        if action == 'B':
            pass
        elif action == 'S':
            # 当R为-的时候，选择S，应该是正奖励
            reward = reward * -1
        else:
            # 选择观望，既不亏损也不会盈利，但会损失机会成本
            # 我们当前对观望的决策持客观态度，reward=0，这
            # 可能需要在不同的大盘行情下适时调整
            reward = 0
            pass
        self.time = s_
        return s_, reward, done
        pass

    def reset(self):
        self.time = 935
        return self.time
        pass

class QLearning:
    #Agent
    def __init__(self, actions, q_table=None, learning_rate=0.01,
                 discount_factor=0.9, e_greedy=0.1):
        self.actions = actions  # action 列表
        self.lr = learning_rate  # 学习速率
        self.gamma = discount_factor  # 折扣因子
        self.epsilon = e_greedy  # 贪婪度
        # 列是action。
        if q_table is None:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)  # Q 表
        else:
            self.q_table = q_table

    # 检测 q_table 中有没有这个 state
    # 如果还没有当前 state, 那我们就插入一组全 0 数据, 作为这个 state 的所有 action 的初始值
    def check_state_exist(self, state):
        # state对应每一行，如果不在Q表中。
        if state not in self.q_table.index:
            # 插入一组全 0 数据，给每个action赋值为0
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # 根据 state 来选择 action
    def choose_action(self, state):
        self.check_state_exist(state)  # 检测此 state 是否在 q_table 中存在
        # 选行为，用 Epsilon Greedy 贪婪方法
        if np.random.uniform() < self.epsilon:
            # 随机选择 action
            action = np.random.choice(self.actions)
        else:  # 选择 Q 值最高的 action
            state_action = self.q_table.loc[state, :]
            # 同一个 state, 可能会有多个相同的 Q action 值, 所以我们乱序一下
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            # 每一行中取到Q值最大的那个
            action = state_action.idxmax()
        return action

    # 学习。更新 Q 表中的值
    def learn(self, s, a, r, s_):
        # s_是下一个状态
        self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_

        # Q(S,A) <- Q(S,A)+a*[R+v*max(Q(S',a))-Q(S,A)]

        q_predict = self.q_table.loc[s, a]  # 根据 Q 表得到的 估计（predict）值

        # q_target 是现实值
        if s_ != 'terminal':  # 下个 state 不是 终止符
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # 下个 state 是 终止符

        # 更新 Q 表中 state-action 的值
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

def update(data, q_table=None):
    env = Market(data)
    RL = QLearning(actions=env.action_space, q_table=q_table)

    for episode in range(100):
        # 初始化 state（状态）
        state = env.reset()

        step_count = 0  # 记录走过的步数

        while True:
            # 更新可视化环境
            # env.render()
            # RL 大脑根据 state 挑选 action
            action = RL.choose_action(str(state))
            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state, reward 和 done (是否到了1500)
            state_, reward, done = env.step(action)
            step_count += 1  # 增加步数
            # 机器人大脑从这个过渡（transition） (state, action, reward, state_) 中学习
            RL.learn(str(state), action, reward, str(state_))
            # 机器人移动到下一个 state
            state = state_
            # 如果时间到了1500, 这回合就结束了，或者是某个止损条件达到了
            if done:
                # print("回合 {} 结束. 总步数 : {}\n".format(episode + 1, step_count))
                break

    # print('模拟交易结束了。')
    # print('\nQ 表:')
    # print(RL.q_table)
    return RL.q_table


def train():
    code = '300750.XSHE'  # 上证指数
    sd = dt.datetime(2020, 10, 1)
    ed = dt.datetime(2021, 1, 19)
    # 我已经把从jqdata读取到了数据存在了本地，这里只是读取出来
    data = pd.read_csv('qlearning_m5.csv', index_col=0)
    data = data.sort_values(['date', 'time'], ascending=False)
    # 计算每根K线收盘时未来三根K线的涨跌幅
    data['R'] = (data.close.shift(3) / data.close - 1) * 100
    data.fillna(0, inplace=True)
    data = data.round({'R': 3})
    data = data.sort_values(['date', 'time'], ascending=True)
    qtb = None
    for k, g in data.groupby(['date']):
        print('train to:', k)
        try:
            # 开始一天一天的训练
            qtb = update(g, qtb)
        except Exception as e:
            print(e)
        print('\nQ 表:')
        print(qtb)
    qtb['time'] = qtb.index
    qtb.to_csv(path_or_buf='qtb({})_{}.csv'.format(code, sd.strftime('%Y_%m_%d')), index=False)
    pass


train()