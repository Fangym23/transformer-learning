#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/19 0:14
# @Author: FANGYIMIN

import matplotlib.pyplot as plt
from NeuralUCB import*
from sklearn.linear_model import LinearRegression


def train_bandit_model(n_bandits=20, n_arms=8, n_features=8, n_samples=100, sigma=0.1):
    """
    Train a Ridge Regression model for a multi-armed bandit problem.

    Parameters:
        n_bandits (int): Number of bandits.
        n_arms (int): Number of arms per bandit (must equal feature dimension).
        n_features (int): Dimension of feature vectors.
        n_samples (int): Number of samples per bandit.
        sigma (float): Standard deviation of Gaussian noise in rewards.

    Returns:
        model (numpy.ndarray): Trained model coefficients for each bandit.
    """
    # Generate bandit data
    bandits = []
    for _ in range(n_bandits):
        # Each bandit has its own true weights
        w_true = np.random.randn(n_features)
        w_true /= np.linalg.norm(w_true)
        source_data.append(w_true*10)

        # Generate orthogonal arm feature vectors
        arms = np.random.randn(n_features, n_arms).T  # Orthogonalized feature vectors

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
        linear_model = LinearRegression( fit_intercept=False)
        linear_model.fit(X, y)
        model.append(linear_model.coef_)

    return np.array(model)


#set global variance
n_sim = 10
round = 1000
regrets = np.empty((3,n_sim,round))
neural_regrets = np.empty((3,n_sim,round))
for i in range(3):
    source_data = []

    # setting of the historical data
    n_bandits = 5*(i+1)
    n_arms = 80
    n_features = 8
    n_samples = 100
    historical_data = train_bandit_model(n_bandits, n_arms, n_features, n_samples)
    source_data = np.array(source_data)

    # setting of the contextual bandit
    T = round
    n_arms = 4
    n_features = 8
    noise_std = 0.1
    confidence_scaling_factor = noise_std
    n_sim = 10
    SEED = 42
    np.random.seed(SEED)

    # setting of the neural network
    p = 0.2
    hidden_size = 32
    epochs = 100
    train_every = 10
    use_cuda = False

    ### mean reward function
    a = np.random.randn(n_bandits)
    a /= np.linalg.norm(a, ord=2)
    reward_func = lambda x: np.dot(a, np.dot(source_data, x)) ** 2

    bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=noise_std, seed=SEED)


    # transfer learning
    for j in range(n_sim):
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
                          historical_data=historical_data,
                          num_his_bandit=n_bandits
                          )

        model.run()
        regrets[i,j, :] = np.cumsum(model.regrets)

    # neuralucb

    for j in range(n_sim):
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
        neural_regrets[i,j, :] = np.cumsum(neural_model.regrets)


# calculate the average
average_regrets = np.mean(regrets, axis=1)  # Shape (3, T)
average_neural_regrets = np.mean(neural_regrets, axis=1)  # Shape (3, T)

t = np.arange(round)

plt.figure(figsize=(12, 7))
colors = {'Transfer': 'blue', 'NeuralUCB': 'red'}
linestyles = ['-', '--', '-.']  # 对应n_bandits=5, 10, 15

for i in range(3):
    n_bandits = 5 * (i + 1)
    plt.plot(t, average_regrets[i],
             color=colors['Transfer'],
             linestyle=linestyles[i],
             linewidth=2,
             label=f'Transfer (n_bandits={n_bandits})')
    plt.plot(t, average_neural_regrets[i],
             color=colors['NeuralUCB'],
             linestyle=linestyles[i],
             linewidth=2,
             label=f'NeuralUCB (n_bandits={n_bandits})')

plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.title('Cumulative Regret Comparison Across Different Configurations', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., fontsize=10)
plt.tight_layout()
plt.savefig('Cumulative Regret Comparison Across Different Configurations.jpg')
plt.show()