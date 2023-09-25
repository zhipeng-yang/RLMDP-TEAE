
import numpy as np
import gym
import pandas as pd
from Uniform_TEAE.Algorithm import DQN as METHOD
import torch


def main():
    # 初始化参数，智能体环境
    Reward = []
    Episode = []
    Step = []
    Num_steps = []
    Mean_Reward = []
    Score = []
    env = gym.make(ENV)
    agent = METHOD(env)
    best_eval_score = -np.inf

    for episode in range(EPISODE):
        # 初始化环境
        total_reward = 0
        Timestep = 0
        state = env.reset()
        action = agent.random_action()
        for step in range(STEP):
            # env.render()
            next_state, reward, done, _ = env.step(action)  # 执行当前动作获得所有转换数据
            total_reward += reward
            agent.perceive(state, action, reward, next_state, done)  # 调用perceive函数存储所有转换数据
            one_hot_action = np.zeros(agent.current_Q_net.action_dim)  # 对 action 进行 one_hot 编码，若选择某个动作，对应位置为1.
            one_hot_action[action] = 1
            one_hot_action = torch.Tensor(one_hot_action)
            state = torch.Tensor(state)
            next_state = torch.Tensor(next_state)
            action = agent.ae_greedy_action(state, one_hot_action, reward, next_state, done)  # 调用epsilon_greedy算法选择动作
            print(f'Episode: {episode:<4}  '
                  f'TIME_STEP: {step:<4}  '
                  f'Return: {total_reward:<5.1f}')
            if done:
                break
            Timestep += 1
            Episode.append(episode)
            Reward.append(total_reward)
            Step.append(Timestep)
            pd_data = pd.DataFrame({
                'EPISODE': Episode,
                'Timestep': Step,
                'Reward': Reward,
                'Algorithm': algorithm, })
            pd_data.to_csv('E:/DESKTOP/TEAE/code/DQN/Uniform_TEAE/data/train/' + ENV + '.csv')

        if episode % params_interval == 0:
            agent.update_target_params()

        if episode % eval_interval == 0:
            total_return = 0
            for i in range(TEST):
                state = env.reset()
                episode_return = 0.0
                done = False
                episode_steps = 0
                while not done:
                    action = agent.action(state)  # 调用action函数,获得目标网络中Q值最大的动作
                    next_state, reward, done, _ = env.step(action)
                    episode_steps += 1
                    if episode_steps + 1 >= STEP:
                        done = True
                    episode_return += reward
                    state = next_state
                total_return += episode_return
            mean_return = total_return / TEST
            if mean_return > best_eval_score:
                best_eval_score = mean_return
            print('-' * 60)
            print(f'Num steps: {Timestep:<5}  '
                  f'Reward: {mean_return:<5.1f}'
                  f'Score: {best_eval_score:<5.1f}')
            print('-' * 60)

            Num_steps.append(Timestep)
            Mean_Reward.append(mean_return)
            Score.append(best_eval_score)
            pd_data2 = pd.DataFrame({
                'Num_steps': Num_steps,
                'Reward': Mean_Reward,
                'Score': Score, })
            pd_data2.to_csv('E:/DESKTOP/TEAE/code/DQN/Uniform_TEAE/data/test/' + ENV + '.csv')


if __name__ == '__main__':
    algorithm = "DQN_Uniform_TEAE"
    EPISODE = 400  # 迭代周期数
    STEP = 5000  # 每个周期迭代时间步
    params_interval = 1
    eval_interval = 2
    TEST = 4
    ENV = "Alien-ram-v0"
    main()
