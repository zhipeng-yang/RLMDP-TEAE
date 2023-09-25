import torch
import torch.nn.functional as F



class MODEL(torch.nn.Module):
    def __init__(self, env):
        super(MODEL, self).__init__()  # 调用父类构造函数
        self.state_dim = env.observation_space.shape[0]  # 状态个数
        # self.action_dim = env.action_space.shape[0]  # 动作个数
        self.action_dim = env.action_space.n  # 动作个数
        # self.action_dim = len(self.action_space)
        # self.state_dim = env.state_dim
        # self.action_dim = env.action_dim
        self.fc1 = torch.nn.Linear(self.state_dim, 20)  # 建立第一层网络 : 随机生成20*4的权重，以及1*20的偏置，Y = XA^T + b
        self.fc1.weight.data.normal_(0, 0.6)  # 设置第一层网络参数，使得第一层网络的权重服从正态分布：均值为0，标准差为0.6
        self.fc2 = torch.nn.Linear(20, self.action_dim)  # 建立第二层网络：随机生成2*20的权重，以及1*2的偏置，Y = XA^T + b

    def create_Q_network(self, x):  # 创建 Q 网络
        x = F.relu(self.fc1(x))  # 调用 torch 的 relu 函数
        Q_value = self.fc2(x)  # 输出 Q_value
        return Q_value

    def forward(self, x, action_input):
        Q_value = self.create_Q_network(x)
        Q_action = torch.mul(Q_value, action_input).sum(
            dim=1)  # 计算执行动作action_input得到的回报。torch.mul:矩阵点乘; torch.sum: dim = 1按行求和，dim = 0按列求和
        return Q_action