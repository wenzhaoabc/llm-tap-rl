# 强化学习在大语言模型训练中的应用

本仓库对常用强化学习方法进行了整理，包括介绍强化学习基本概念和示例代码，概念涉及强化学习中的Q-learning、DQN，策略梯度，广义优势估计，重要性采样，近端策略优化等。

补充了大语言模型后训练中的相关RL方法。包括PPO，DPO，GRPO及GRPO算法的相关变体。

## 章节目录：

- `0.simple_rl.py`：最基础的强化学习示例，用一个最小环境说明状态、动作、奖励和策略更新的基本概念，适合先建立直觉。
- `1.Qlearning.py`：Q-learning 示例，展示值函数表格更新、探索与利用、以及离散状态动作空间下的经典强化学习流程。
- `2.dqn.ipynb`：DQN 入门 notebook，介绍如何用神经网络近似 Q 函数，以及经验回放、目标网络等关键技巧。
- `2.DoubleDQN.ipynb`：Double DQN notebook，对比普通 DQN，说明如何缓解动作价值高估问题。
- `3.reward_model.ipynb`：奖励模型训练相关内容，介绍偏好数据、pairwise loss，以及 RLHF 中 reward model 的基本作用。
- `4.reinforce.ipynb`：REINFORCE 策略梯度算法，从最基本的 Monte Carlo policy gradient 出发，解释策略优化的梯度形式。
- `5.actor_critic.ipynb`：Actor-Critic notebook，展示策略网络和价值网络联合训练的基本框架，为 PPO 等算法做铺垫。
- `6.ppo.ipynb`：PPO notebook，介绍 clipped objective、KL 约束、GAE 等核心内容，是从经典 RLHF 过渡到 LLM 后训练的重要一章。
- `7.grpo.ipynb`：GRPO notebook，介绍 Group Relative Policy Optimization 的基本思想、组内相对优势估计，以及它与 PPO 的区别。
- `8.LLM-RL.ipynb`：面向大语言模型后训练的 RL 方法综述，主要整理 PPO、DPO、GRPO 在 TRL 中的原理和最小实现方式。
- `9.GRPO-Variants.ipynb`：GRPO 相关变体专题，按“改了 GRPO 的哪一层”来分类讲解 DAPO、Dr.GRPO、BNPO、CISPO、GSPO、GFPO、GAPO 等方法，并补充 TRL 支持情况与示意实现。

推荐阅读顺序：

1. `0.simple_rl.py` 和 `1.Qlearning.py`，建立最基本的强化学习直觉。
2. `2.dqn.ipynb`、`2.DoubleDQN.ipynb`、`4.reinforce.ipynb`、`5.actor_critic.ipynb`，补齐值函数和策略梯度两条主线。
3. `6.ppo.ipynb` 和 `7.grpo.ipynb`，理解大模型后训练里最关键的在线优化方法。
4. `8.LLM-RL.ipynb` 和 `9.GRPO-Variants.ipynb`，把 RLHF、TRL 实现和 GRPO 变体联系起来。

## 参考资料

参考书目包括：

- [大规模语言模型：从理论到实践](https://intro-llm.github.io/)
- [动手学强化学习](https://hrl.boyuai.com/)
- [蘑菇书EasyRL](https://datawhalechina.github.io/easy-rl/#/)
- [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v2)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325v3)

参考博客

- [深度强化学习之深度Q网络DQN详解](https://zhuanlan.zhihu.com/p/145102068)
- [ChatGPT原理详解+实操(1)----SFT(GPT模型精调)](https://zhuanlan.zhihu.com/p/609795142)
- [ChatGPT原理详解+实操(2)----RM(reward model)](https://zhuanlan.zhihu.com/p/610147705)
- [复旦NLP组开源PPO-Max](https://www.51cto.com/article/761044.html)
- [想训练ChatGPT？得先弄明白Reward Model怎么训（附源码）](https://mp.weixin.qq.com/s/1v4Uuc1YAZ9MRr1UWMH9xw)
- [LLM常见问题（强化学习部分）](https://juejin.cn/post/7302993899106713600)
- [KL散度(Kullback-Leibler Divergence)介绍及详细公式推导](https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/)
- [初学机器学习：直观解读KL散度的数学概念](https://www.jiqizhixin.com/articles/2018-05-29-2)
- ...

## 许可证

- 本仓库采用 MIT License 开源，详见 [LICENSE](LICENSE)。
