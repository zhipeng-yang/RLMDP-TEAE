import random
import torch
import numpy as np
from PER_TEAE.Model import MODEL
from PER_TEAE.Sample import Memory
import math

# 设置参数
GAMMA = 0.99  # 折现因子
# GAMMA = 0.009  # 折现因子
INITIAL_EPSILON = 1  # 初始的epsilon
# INITIAL_EPSILON = 0.5  # 初始的epsilon
# FINAL_EPSILON = 0.01  # 最终的epsilon
REPLAY_SIZE = 10000  # 经验池大小
BATCH_SIZE = 128  # Minimatch 大小
# Update_Target_Freq = 10  # 目标网络参数更新频率
Bate = 0.5


class DQN:
    def __init__(self, env):
        self.action_dim = env.action_space.n
        self.replay_total = 0  # 定义回放次数
        self.target_Q_net = MODEL(env)  # 定义目标网络
        self.current_Q_net = MODEL(env)  # 定义当前网络
        self.memory = Memory(capacity=REPLAY_SIZE)  # 定义经验池大小
        self.time_step = 0  # 定义时间步数
        self.epsilon = INITIAL_EPSILON  # 定义初始epsilon
        self.optimizer = torch.optim.Adam(params=self.current_Q_net.parameters(), lr=0.00025)  # 使用Adam优化器

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))  # np.hstack: 按水平方向堆叠数组构成一个新的数组
        self.memory.store(transition)  # 调用store函数存储转换关系数据

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.current_Q_net.action_dim)  # 对 action 进行 one_hot 编码，若选择某个动作，对应位置为1.
        one_hot_action[action] = 1
        self.store_transition(state, one_hot_action, reward, next_state, done)
        self.replay_total += 1  # 完成一次存储，回放次数加1
        if self.replay_total > BATCH_SIZE:  # 判断回放总次数是否大于BATCH_SIZE，大于就开始训练
            self.train_Q_network()

    def train_Q_network(self):  # 定义训练
        self.time_step += 1
        # 1. 从经验池采样
        tree_idx, minibatch, ISWeights, p_yxj = self.memory.sample(BATCH_SIZE)  # 调用sample函数采样
        state_batch = torch.tensor(minibatch[:, 0:self.current_Q_net.state_dim], dtype=torch.float32)
        # print('状态:', state_batch.detach().numpy())
        # 取出state_batch，minibatch中所有行的前4列
        action_batch = torch.tensor(
            minibatch[:, self.current_Q_net.state_dim:self.current_Q_net.state_dim + self.current_Q_net.action_dim],
            dtype=torch.float32)
        # 取出action_batch，minibatch中所有行的第5，6列
        reward_batch = [data[self.current_Q_net.state_dim + self.current_Q_net.action_dim] for data in minibatch]
        # 取出reward_batch，minibatch中每一行的第7列
        next_state_batch = torch.tensor(minibatch[:,
                                        self.current_Q_net.state_dim + self.current_Q_net.action_dim + 1: 2 * self.current_Q_net.state_dim + self.current_Q_net.action_dim + 1],
                                        dtype=torch.float32)
        # 取出next_state_batch，minibatch中所有行的第8，9，10，11列

        # 2. 计算 y
        y_batch = []  # 定义y_batch为一个数组
        Q_value_batch = self.target_Q_net.create_Q_network(next_state_batch)  # 调用 create_Q_network，使用target_Q_net计算Q值
        max_target_Q_value_batch = torch.max(Q_value_batch, dim=1)[0]  # 返回每一行中的最大值
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][
                2 * self.current_Q_net.state_dim + self.current_Q_net.action_dim + 1]  # 取出 minibatch 中每一行的第12个数据，即取出是否到达终止的标识
            if done:
                y_batch.append(reward_batch[i])  # 若到达终止条件，y_batch=reward_batch
            else:  # 若未到达终止条件
                max_target_Q_value = max_target_Q_value_batch[i]  # 取出在目标网络中每个状态执行动作获得的最大Q值
                y_batch.append(reward_batch[i] + GAMMA * max_target_Q_value)  # 计算Y, reward_batch + GAMMA *目标网络中动作的最大Q值
        y = self.current_Q_net(torch.FloatTensor(state_batch),
                               torch.FloatTensor(action_batch))  # 调用当前网络计算在state_batch下执行action_batch得到的回报
        # torch.FloatTensor ：转换数据类型为32位浮点型
        y_batch = torch.FloatTensor(y_batch)
        cost = self.loss(y_batch, y, torch.tensor(ISWeights))  # 调用loss函数，计算损失函数
        self.optimizer.zero_grad()  # 初始化，把梯度置零，把loss关于weight的导数变成0.
        cost.backward()  # 计算梯度
        self.optimizer.step()  # 根据梯度更新参数
        abs_errors = torch.abs(y_batch - y)  # 计算y_batch - y的绝对值
        self.memory.batch_update(tree_idx, abs_errors)  # 调用batch_update函数，更新树

    @staticmethod
    def loss(y_output, y_true, ISWeights):  # 定义损失函数
        value = y_output - y_true
        return torch.mean(value * value * ISWeights)

    @staticmethod
    def TD_error(y_output, y_true):  # 定义损失函数
        value = abs(y_output - y_true)
        return value

    # def e_greedy_action(self, state):  # 定义epsilon_greedy算法
    #     Q_value = self.current_Q_net.create_Q_network(torch.FloatTensor(state))  # 跟据输入状态调用当前网络计算 Q_value
    #     if random.random() <= self.epsilon:  # 使用random函数随机生成一个0-1的数，若小于epsilon，更新epsilon并返回随机动作
    #         self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
    #         return random.randint(0, self.current_Q_net.action_dim - 1)
    #     else:  # 否则更新 epsilon， 并返回 Q_value 最大时对应的动作
    #         self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
    #         return torch.argmax(Q_value).item()  # 返回Q_value中最大值的索引值

    def ae_greedy_action(self, state, action, reward, next_state, done):  # 定义epsilon_greedy算法
        y = self.current_Q_net(torch.FloatTensor(state), torch.FloatTensor(action))

        current_a = self.current_Q_net.create_Q_network(next_state)
        max_current_action = torch.argmax(current_a, dim=-1)
        Q_value = self.target_Q_net.create_Q_network(next_state)
        if done:
            y_ = reward
        else:
            y_ = reward + GAMMA * Q_value[max_current_action]

        abs_td = self.TD_error(y, y_)
        W_f = (1 - math.exp(-abs_td/Bate)) / (1 + math.exp(-abs_td/Bate))

        Q_value_ = self.current_Q_net.create_Q_network(torch.FloatTensor(state))
        if random.random() <= self.epsilon:  # 使用random函数随机生成一个0-1的数，若小于epsilon，更新epsilon并返回随机动作
            self.epsilon = (1 / self.action_dim) * W_f + (1 - 1 / self.action_dim) * self.epsilon
            return random.randint(0, self.current_Q_net.action_dim - 1)
        else:  # 否则更新 epsilon， 并返回 Q_value 最大时对应的动作
            self.epsilon = (1 / self.action_dim) * W_f + (1 - 1 / self.action_dim) * self.epsilon
            # print("001", self.epsilon)
            return torch.argmax(Q_value_).item()  # 返回Q_value中最大值的索引值

    def random_action(self):  # 定义随机动作
        return random.randint(0, self.current_Q_net.action_dim - 1)

    def action(self, state):  # 返回目标网络中Q_value最大值的索引值
        return torch.argmax(self.target_Q_net.create_Q_network(torch.FloatTensor(state))).item()

    def update_target_params(self):  # 更新目标网络参数
        torch.save(self.current_Q_net.state_dict(), 'net_params.pkl')  # 保存当前网络参数到本地
        self.target_Q_net.load_state_dict(torch.load('net_params.pkl'))  # 上传当前网络参数并赋给目标网络
