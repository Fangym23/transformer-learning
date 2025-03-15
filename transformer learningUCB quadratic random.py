#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/15 13:53
# @Author: FANGYIMIN


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/11 16:50
# @Author: FANGYIMIN
import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import orth
from neural_exploration import *
import matplotlib.pyplot as plt
from NeuralUCB import*


#setting of the historical data
n_bandits = 5
n_arms = 8
n_features = 8
n_samples = 100
historical_data = np.random.rand(5, 8)


#setting of the contextual bandit
T = int(5e2)
n_arms = 4
n_features = 8
noise_std = 0.1
confidence_scaling_factor = noise_std
n_sim = 2
SEED = 42
np.random.seed(SEED)


#setting of the neural network
p = 0.2
hidden_size = 32
epochs = 100
train_every = 10
use_cuda = False


### mean reward function
a = np.random.randn(n_features)
a /= np.linalg.norm(a, ord=2)
reward_func = lambda x: 100*np.dot(a, x)**2

bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=noise_std, seed=SEED)

regrets = np.empty((n_sim, T))

#transfer learning
for i in range(n_sim):
    bandit.reset_rewards()
    model = NeuralUCB(bandit,
                      hidden_size=hidden_size,
                      reg_factor=1.0,
                      delta=0.1,
                      confidence_scaling_factor=confidence_scaling_factor,
                      training_window=100,
                      p=p,
                      learning_rate=0.01,
                      epochs=epochs,
                      train_every=train_every,
                      use_cuda=use_cuda,
                      historical_data = historical_data,
                      num_his_bandit = n_bandits
                      )

    model.run()
    regrets[i] = np.cumsum(model.regrets)

#neuralucb
neural_regrets = np.empty((n_sim, T))

for i in range(n_sim):
    bandit.reset_rewards()
    neural_model = NeuralUCB_previous(bandit,
                      hidden_size=hidden_size,
                      reg_factor=1.0,
                      delta=0.1,
                      confidence_scaling_factor=confidence_scaling_factor,
                      training_window=100,
                      p=p,
                      learning_rate=0.01,
                      epochs=epochs,
                      train_every=train_every,
                      use_cuda=use_cuda,
                     )
    neural_model.run()
    neural_regrets[i] = np.cumsum(neural_model.regrets)


fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)

t = np.arange(T)

mean_regrets = np.mean(regrets, axis=0)
neural_mean_regrets = np.mean(neural_regrets,axis=0)
std_regrets = np.std(regrets, axis=0) / np.sqrt(regrets.shape[0])
std_neural = np.std(neural_regrets,axis=0)/np.sqrt(neural_regrets.shape[0])
# transfer learning
ax.plot(t, mean_regrets, label='transfer learning')
ax.fill_between(t, mean_regrets - 2 * std_regrets, mean_regrets + 2 * std_regrets, alpha=0.15)

# 绘制第二条线：neural_mean_regrets
ax.plot(t, neural_mean_regrets, label='NeuralUCB')
ax.fill_between(t, neural_mean_regrets - 2 * std_neural, neural_mean_regrets + 2 * std_neural, alpha=0.15)

ax.set_title('Cumulative Regret')
ax.legend()  # 添加图例

plt.tight_layout()
plt.savefig('transfer learning and NeuralUCB_quad.jpg')
plt.show()

