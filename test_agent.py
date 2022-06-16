"""
Tests a trained RL agent
"""
import os
import numpy as np
from burgers_obj import *
from env import StochasticBurgersEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

def plot_compared_models(t, x, model_traj, em_traj, rewards):
    # plot the spatial results of model trajectory
    fig, axs = plt.subplots(3)
    axs[0].plot(x, np.transpose(model_traj))
    axs[1].plot(x, np.transpose(em_traj))
    axs[2].plot(t, rewards)
    fig.suptitle('Trained agent vs uncontrolled equation and Rewards')
    plt.show()
    pass

def test(model):
    u = BurgersVelocity(dt=0.0001, dx=0.1)
    env = StochasticBurgersEnv(u=u)

    traj, rewards = [], []

    obs = env.reset()
    for i in range(u.max_steps):
        act, _states = model.predict(obs)
        obs, reward, done, info = env.step(act)
        traj.append(obs)
        rewards.append(reward)

    em_traj = u.euler_maruyama()

    plot_compared_models(u.T_range, u.X_range, traj, em_traj, rewards)

if __name__ == "__main__":
    model = PPO.load("models/sbe_rl_model_5.zip")
    test(model)
