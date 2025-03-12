#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/11 16:50
# @Author: FANGYIMIN
import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import orth
from neural_exploration import *
import matplotlib.pyplot as plt
from NeuralUCB_previous import*


def train_bandit_model(n_bandits=20, n_arms=8, n_features=8, n_samples=100, sigma=0.1, alpha=1.0):
    """
    Train a Ridge Regression model for a multi-armed bandit problem.

    Parameters:
        n_bandits (int): Number of bandits.
        n_arms (int): Number of arms per bandit (must equal feature dimension).
        n_features (int): Dimension of feature vectors.
        n_samples (int): Number of samples per bandit.
        sigma (float): Standard deviation of Gaussian noise in rewards.
        alpha (float): Regularization strength for Ridge Regression.

    Returns:
        model (numpy.ndarray): Trained model coefficients for each bandit.
    """
    # Generate bandit data
    bandits = []
    for _ in range(n_bandits):
        # Each bandit has its own true weights
        w_true = np.random.randn(n_features)
        w_true /= np.linalg.norm(w_true)

        # Generate orthogonal arm feature vectors
        arms = orth(np.random.randn(n_features, n_arms)).T  # Orthogonalized feature vectors

        bandits.append({'w_true': w_true, 'arms': arms})

    # Data collection and training
    model = []
    for i, bandit in enumerate(bandits):
        X = []
        y = []
        arms = bandit['arms']
        w_true = bandit['w_true']

        # Generate sample data
        for _ in range(n_samples):
            arm_idx = np.random.choice(n_arms)
            x = arms[arm_idx]
            mu = 10*np.dot(x, w_true)
            reward = np.random.normal(mu, sigma)  # Add Gaussian noise
            X.append(x)
            y.append(reward)

        X = np.array(X)
        y = np.array(y)

        # Train Ridge Regression model
        ridge_model = Ridge(alpha=alpha, fit_intercept=False)
        ridge_model.fit(X, y)
        model.append(ridge_model.coef_)

    return np.array(model)


#setting of the historical data
n_bandits = 20
n_arms = 8
n_features = 8
n_samples = 100
historical_data = train_bandit_model(n_bandits, n_arms, n_features, n_samples)



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
plt.savefig('transfer learning and NeuralUCB_quad1.jpg')
plt.show()

