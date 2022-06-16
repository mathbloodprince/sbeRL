"""
Creates a Burgers Velocity field object to interact with the environment
"""
import numpy as np
import matplotlib.pyplot as plt


class BurgersVelocity(object):

    def __init__(self, dt, dx):
        self.dt = dt
        self.dx = dx
        self.t0, self.max_T = 0.0, 2.0
        self.max_steps = int((self.max_T/self.dt)+1)
        self.a, self.b = 0.0, 2*np.pi
        self.n = int(self.b/self.dx)

        self.T_range, self.X_range = self.discretize()

        self.u0 = self.phi(self.X_range)
        self.u = self.u0
        self.f = np.zeros((self.n-1,))
        pass

    # initializes the velocity field w sin function
    def phi(self, x):
        u0 = [0.0]
        for k in range(1, len(x)-1):
            u0.append(np.sin(x[k]))
        u0.append(0.0)
        return u0

    def discretize(self,):
        T_range = [self.t0 + k*self.dt for k in range(self.max_steps)]
        X_range = [self.a]
        for k in range(1, self.n):
            X_range.append(self.a + k*self.dx)
        X_range.append(self.b)
        return T_range, X_range

    def set_action(self, f):
        self.f = f

    def get_action(self,):
        return self.f

    def set_state(self, u):
        self.u = u

    def get_state(self,):
        return self.u

    def forward(self,):
        # get the current state
        ut = self.get_state()

        # Euler-Maruyama for one time step
        u_next = np.zeros((self.n+1,))
        u_next[0] = 0.0

        self.D, self.H, self.F = self.set_matricies(self.X_range, self.f)
        b = self.A(ut[1:self.n], self.D, self.H, self.F, self.n)
        sigma = np.sqrt(self.n)
        W = np.array([np.random.normal(0, self.dt) for k in range(self.n-1)])
        u_next[1:self.n] = ut[1:self.n] + np.multiply(b, self.dt) + np.multiply(sigma, W)
        u_next[self.n] = 0.0

        # update the next state
        self.set_state(u_next)

        return u_next

    def A(self, x, D, H, F, n):
        return np.square(n)*np.dot(D, x) + F + np.multiply(n, H) 

    def set_matricies(self, x, f):
        D = np.zeros((self.n-1, self.n-1))
        H, F = [], []

        # set D matrix (n-1, n-1)
        for i in range(len(D)):
            for j in range(len(D[0])):
                if i == j:
                    D[i, j] = -2
                elif abs(i - j) > 1:
                    D[i, j] = 0
                elif abs(i - j) == 1:
                    D[i, j] = 1

        # set H and F matricies: both (n-1, )
        for k in range(1, (len(x)-1)):
            Hk = (1/6)*(x[k+1]**2 - x[k-1]**2 + x[k+1]*x[k] - x[k]*x[k-1])
            H.append(Hk)
            
            Fk = f[k-1]
            F.append(Fk)
        return D, H, F

    # computes integral mean over spatial domain of full trajectory U
    def compute_target(self, U):
        return (1/(2*np.pi))*np.mean(np.trapz(U, axis=1))

    # euler_maruyama scheme for SDE solving
    def euler_maruyama(self,):
        u = [self.u0]
        for t in range(1, len(self.T_range)):
            u_next = np.zeros((self.n+1,))
            u_next[0] = 0.0
            self.D, self.H, self.F = self.set_matricies(self.X_range, f=np.zeros((self.n-1,)))
            b = self.A(u[t-1][1:self.n], self.D, self.H, self.F, self.n)
            sigma = np.sqrt(self.n)
            W = np.array([np.random.normal(0, self.dt) for k in range(self.n-1)])
            u_next[1:self.n] = u[t-1][1:self.n] + np.multiply(b, self.dt) + np.multiply(sigma, W)
            u_next[self.n] = 0.0
            u.append(u_next)
        return u

    def reset(self,):
        u = self.u0
        f = np.zeros(self.n-1,)
        self.set_state(u)
        self.set_action(f)
        pass

def plot_spatial(X, U):
    plt.plot(X, U)
    plt.show()
    pass

if __name__ == "__main__":
    dt, dx = 0.0001, 0.1 
    u = BurgersVelocity(dt, dx)
    U = u.euler_maruyama()
    plot_spatial(u.X_range, np.transpose(U))
