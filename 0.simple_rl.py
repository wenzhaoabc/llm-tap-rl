"""
简单的强化学习模型，随机策略选择
"""
import gymnasium as gym
import numpy as np

# 创建 FrozenLake 环境，指定 is_slippery 为 False 使环境变得确定性，即去除滑动效应
# 如果你想要原始的随机滑动效应，可以设置为 True
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# 初始化变量
num_episodes = 1000  # 总共运行的游戏回合数
max_steps_per_episode = 100  # 每个回合的最大步数

# 运行多个回合来展示随机策略的效果
for episode in range(num_episodes):
    # 重置环境状态到初始位置
    state = env.reset()
    done = False  # 游戏是否结束的标志
    total_rewards = 0  # 累计奖励

    for step in range(max_steps_per_episode):
        # 在环境中采取随机行动
        action = env.action_space.sample()

        # 执行行动，得到一些反馈信息
        new_state, reward, done, truncated, info = env.step(action)

        # 累加奖励
        total_rewards += reward

        # 如果游戏结束，跳出循环
        if done:
            break

        # 更新状态
        state = new_state

    # 打印本回合的信息
    print(
        f"Episode: {episode + 1}, Total reward: {total_rewards}, Final state: {state}"
    )

# 关闭环境
env.close()
