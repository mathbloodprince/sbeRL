"""
Carries out the process of training our agent
"""
import numpy as np
from vec_env_utils import make_vec_env
from stable_baselines3 import PPO
from burgers_obj import *
from env import StochasticBurgersEnv

def train():
    # create velo object
    u = BurgersVelocity(dt=0.0001, dx=0.1)
    vec_env = make_vec_env(u=u, nenv=10)
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space

    # train agent -> change policy in future?
    model = PPO(policy="MlpPolicy", env=vec_env)
    model.learn(total_timesteps=1000)

    vec_env.reset()
    vec_env.close()

    model.save("models/sbe_rl_model_5")

    pass

if __name__ == "__main__":
    train()

