"""
基于学习价值的智能体，使用Q-learning来学习价值函数
"""

import gymnasium as gym
import numpy as np

# 创建FrozenLake环境
# 不渲染图形
env = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    desc=[
        "SFF",
        "FFH",
        "HFG",
    ],  # custom map, Start, Frozen, Hole, Gole
    map_name="3x3",  # map size
    # render_mode="human",
)

# Q表初始化，所有Q值设为0
# 特定[state, action]下的奖励
print(env.observation_space.n, env.action_space.n)
Q = np.full((env.observation_space.n, env.action_space.n), 0, dtype=np.float64)
print(Q.shape)

# 超参数设置
learning_rate = 0.9  # 学习率
discount_factor = 0.5  # 折扣因子
num_episodes = 2000  # 训练次数


# 用于动态展示训练进度
def print_progress(episode):
    print(f"Episode: {episode}\r")
    if episode % 10 == 0:
        print(Q)


# Q-learning训练过程
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # 从Q表中选择具有最高Q值的动作，或者随机选择动作
        if np.random.rand() < (1 / (episode + 1)):
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(Q[state, :])  # 利用，贪婪策略

        # 执行动作，观察新状态和奖励
        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            if new_state == 4 * 4 - 1:
                reward += 5
            else:
                reward -= 5
        else:
            reward = 0
        # print("reward = ", reward)
        # 更新Q表
        # 采用时序差分的方法更新Q表，奖励高的(状态，动作)对前后的值也会得到强化
        # https://datawhalechina.github.io/easy-rl/#/chapter3/chapter3?id=_332-%e6%97%b6%e5%ba%8f%e5%b7%ae%e5%88%86
        # https://zhuanlan.zhihu.com/p/145102068
        # learning_rate和discount_factor的值都介于0-1
        # learning_rate用于平衡旧Q值与新Q值的权重
        # discount_factor用于平衡当前奖励与远期奖励，np.max(Q[new_state, :])可以理解为new_state的状态价值，即
        # state状态下预期奖励奖励的期望
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[new_state, :])
        )

        state = new_state

    # 打印训练进度

    print_progress(episode)


print("\nTraining finished.\n")
print(Q)

# 展示智能体的表现
# 重建环境，渲染图形
env = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    desc=[
        "SFF",
        "FFH",
        "HFG",
    ],  # custom map, Start, Frozen, Hole, Gole
    map_name="3x3",  # map size
    render_mode="human",
)

Q[0, 2] = 100
Q[1, 1] = 100
Q[4, 1] = 100
Q[7, 2] = 100


for episode in range(3):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    print(f"Episode {episode + 1}")

    while not (terminated or truncated):
        env.render()  # 渲染环境的当前状态
        action = np.argmax(Q[state, :])  # 选择最佳动作
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished with a total reward of: {total_reward}")
            break


env.close()
