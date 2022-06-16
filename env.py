import gym
from gym import spaces
import numpy as np
from burgers_obj import BurgersVelocity

class StochasticBurgersEnv(gym.Env):

    def __init__(self, u):
        self.u = u # burgers velocity object

        # currently observation consists only of u(t, x) <-> add time as a parameter
        obs_shape = (self.u.n+1,)
        act_shape = (self.u.n-1,)

        # might have to alter observations to for Dirichlet bdry conditions
        self.observation_space = spaces.Box(-2.5, 2.5, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, shape=act_shape, dtype=np.float32)

        self.step_count = 0
        self.max_steps = self.u.max_steps # max number of time steps

        self.lbda = 0.2
        pass

    # compute reward at each time step? over full horizon?
    def compute_reward(self, u, f):
        self.full_traj = self.u.euler_maruyama()
        u_star = self.u.compute_target(self.full_traj)
        rt = -0.5*(np.trapz(np.square(u-u_star)) + 0.5*self.lbda*np.square(np.linalg.norm(f)))
        return rt

    def step(self, f):
        self.step_count += 1

        self.u.set_action(f)
        u_next = self.u.forward()
        reward = self.compute_reward(u_next, f)

        done = False
        if self.step_count > self.max_steps:
            done = True
        info = dict(ct=self.step_count)

        return u_next, reward, done, info

    def reset(self,):
        self.u.reset()
        u = self.u.get_state()
        return u
