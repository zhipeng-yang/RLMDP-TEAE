
import numpy as np


class SumTree(object):  # 定义SumTree
    data_pointer = 0  # 初始化数据指针

    def __init__(self, capacity):
        self.capacity = capacity  # 定义容量
        self.tree = np.zeros(2 * capacity - 1)  # 初始化树结构,用来存储优先级
        self.data = np.zeros(capacity, dtype=object)  # 初始化数据结构,用来存储所有转换关系的数据

    def add(self, p, data):  # 更新数据
        tree_idx = self.data_pointer + self.capacity - 1  # 初始化树节点的索引
        self.data[self.data_pointer] = data  # 更新,存储数据
        self.update(tree_idx, p)  # 调用 update 函数,更新优先级 P
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # 若指针大于等于SumTree的容量,重置为 0
            self.data_pointer = 0

    def update(self, tree_idx, p):  # 更新优先级
        change = p - self.tree[tree_idx]  # 计算变化量
        self.tree[tree_idx] = p  # 存储优先级
        while tree_idx != 0:  # 若树节点的索引不为 0 ,更新父节点
            tree_idx = (tree_idx - 1) // 2  # 寻找父节点
            self.tree[tree_idx] += change  # 更新父节点

    def get_leaf(self, v):  # 采样
        parent_idx = 0  # 初始化父亲节点索引
        while True:
            cl_idx = 2 * parent_idx + 1  # 计算左右两个孩子的索引
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # 若左孩子索引大于树的长度,那么父亲节点索引就是叶节点索引
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:  # 若 V 小于等于左孩子,那么就把左孩子的索引赋给父亲节点索引
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]  # 否则用 V - 左孩子, 并把右孩子的索引赋给父亲节点索引
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1  # 计算数据索引
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]  # 返回叶子节点索引,对应存储的优先级\转换关系的数据

    @property  # 装饰器,把一个方法变成属性调用的
    def total_p(self):
        return self.tree[0]  # 返回 total_p


class Memory(object):
    epsilon = 0.01  # 设置阈值,防止出现 0 的优先级
    alpha = 0.6  # [0~1] 决定我们要使用多少 Importance Sampling weight 的影响, 如果 alpha = 0, 我们就没使用到任何 Importance Sampling.
    beta = 0.4  # importance sampling 系数, 从 0.4 到 1
    beta_increment_per_sampling = 0.001  # 重要性每次增长0.001
    abs_err_upper = 1.  # 预设 P 的最大值

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):  # 存储转换关系的数据到 SumTree
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # 取出 SumTree中倒数capacity个值中的最大的一个,也就是最大的叶子节点值
        if max_p == 0:  # 若 max_p为0,则将 abs_err_upper 赋给 max_p
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # 调用add函数更新 SumTree 的数据

    def sample(self, n):  # 采样 n 个样本
        b_idx, p_yxj, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=np.int32), np.empty(
            (n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment 优先级分段
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # 更新 beta, importance sampling 系数,最大为1
        min_prob = np.min(
            self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # min_prob: 用叶子节点中最小的优先级值除以总的优先级值 (方便后边计算重要性采样权重)
        if min_prob == 0:  # 若 min_prob 为 0,给它赋值为 0.0001 (后边计算,这个出现在分母,不能为0)
            min_prob = 0.00001
        for i in range(n):  # 遍历 n
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)  # 在 a,b 之间随机生成一个数
            idx, p, data = self.tree.get_leaf(v)  # 调用 get_leaf 函数,返回叶子节点索引,对应存储的优先级\转换关系的数据
            prob = p / self.tree.total_p  # 计算优先级概率
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)  # 计算重要性采样权重
            b_idx[i], b_memory[i, :], p_yxj[i] = idx, data, p  # 存储索引，所有转换关系数据
        # print("abcd:", p_yxj)
        return b_idx, b_memory, ISWeights, p_yxj  # 返回索引，数据，权重

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # 防止出现 0
        # print("TD error", abs_errors)
        clipped_errors = np.minimum(abs_errors.detach().numpy(), self.abs_err_upper)  # 比较两个数组并返回一个包含元素最小值的新数
        ps = np.power(clipped_errors, self.alpha)  # p**alpha
        for ti, p in zip(tree_idx, ps):  # ti,p在 zip(tree_idx, ps) 中取，zip()是一个打包函数，将 tree_idx, ps 打包。
            self.tree.update(ti, p)  # 调用update函数，更新优先级
