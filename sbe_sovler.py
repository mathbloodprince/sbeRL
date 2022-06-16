import numpy as np
import matplotlib.pyplot as plt

# forcing function: 0.0 is no forcing
def f(xk):
    return -10.0

# returns the (n-1) dimensional vector field for EM scheme <-> b(t, u(t)) coefficient
def A(x, D, H, F, n):
    return np.square(n)*np.dot(D, x) + F + np.multiply(n, H)

def set_matricies(x, n):
    D = np.zeros((n-1, n-1))
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

        Fk = f(x[k])
        F.append(Fk)

    return D, H, F

# initial function
def phi(x):
    u0 = [0.0]
    for k in range(1, len(x)-1):
        u0.append(np.sin(x[k]))
    u0.append(0.0)
    return u0

# euler_maruyama scheme for SDE solving
def euler_maruyama(u0, T_range, X_range, D, H, F, n):
    u = [u0]
    for t in range(1, len(T_range)):
        u_next = np.zeros((n+1, ))
        u_next[0] = 0.0
        b = A(u[t-1][1:n], D, H, F, n)
        sigma = np.sqrt(n)
        dt = T_range[t]-T_range[t-1]
        W = np.array([np.random.normal(0, dt) for k in range(n-1)])
        u_next[1:n] = u[t-1][1:n] + np.multiply(b, dt) + np.multiply(sigma, W)
        u_next[n] = 0.0
        u.append(u_next)
    return u

# discretizes the time and spatial domains of the SPDE
def discretize(dt, dx, n):
    t0, T = 0.0, 1.0
    T_range = [t0 + k*dt for k in range(int(T/dt) + 1)]
    a, b = 0.0, 2*np.pi
    X_range = [a]
    for k in range(1, n):
        X_range.append(a + k*dx)
    X_range.append(b)
    return T_range, X_range


def plot_spatial(X_range, U):
    plt.plot(X_range, U)
    plt.show()

if __name__ == "__main__":
    dt, dx = 0.0001, 0.1
    n = int(2*np.pi/dx) # number of partition pts/process dimension

    T_range, X_range = discretize(dt, dx, n)

    D, H, F = set_matricies(X_range, n)
    u0 = phi(X_range)

    u = euler_maruyama(u0, T_range, X_range, D, H, F, n)

    plot_spatial(X_range, np.transpose(u[50]))
