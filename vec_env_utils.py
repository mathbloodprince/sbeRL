from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from env import StochasticBurgersEnv
from copy import deepcopy


class EnvMaker:
    def __init__(self, u):
        self.u = u

    def __call__(self,):
        u = deepcopy(self.u)
        env = StochasticBurgersEnv(u)
        return env

def make_vec_env(u, nenv):
    return VecMonitor(SubprocVecEnv([EnvMaker(u) for i in range(nenv)]))

