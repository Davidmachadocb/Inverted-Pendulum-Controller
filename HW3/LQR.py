import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from visual import plot_results, animate_cart_pendulum

# Define system constants
M: float = 1.0      # Mass of the cart (kg)
L: float = 0.56     # Length of the pendulum (m)
G: float = 9.81     # Acceleration due to gravity (m/s^2)
D: float = 0.1      # Damping coefficient
m: float = 0.356    # Mass of the point (kg)

A = np.array([
    [0, 1, 0, 0],
    [G / L, -D / L, 0, 0],
    [0, 0, 0, 1],
    [(m * G) / (m + M), (-m * D) / (m + M), 0, 0]
])

B = np.array([
    [0],
    [-1 / L],
    [0],
    [(1 - m) / (m + M)]
])

# weighting matrices Q and R
Q = np.diag([30, 1, 35, 1])
R = np.array([[50]])

# Algebraic Riccati Equation
S = linalg.solve_continuous_are(A, B, Q, R)

# gain K
K = np.linalg.inv(R) @ B.T @ S

t_span = (0, 25) 
dt = 0.05
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
s0 = np.array([np.pi/3, 0, 0, 0])

# Track control actions
control_actions = np.zeros(len(t_eval))
eval_index = 0

'''def setpoint(t):
    return (t//6)'''

def lqr_control(t, state):
    global control_actions, eval_index
    ds = np.zeros_like(state)
    _th, _th_d, _x, _x_d = state

    u = -K @ (state - [0, 0, 0, 0])

    damping_term = D * _th_d

    ds[0] = _th_d
    ds[1] = (G * _th - u - damping_term) / L
    ds[2] = _x_d
    ds[3] = (u-ds[1]*M*L) / (M + m)

    while eval_index < len(t_eval) and t >= t_eval[eval_index]:
        control_actions[eval_index] = u[0]
        eval_index += 1

    return ds

solution = solve_ivp(lqr_control, t_span, s0, t_eval=t_eval, method='RK45')

plot_results(t_eval, solution=solution.y.T, control_actions=control_actions)
#animate_cart_pendulum(solution.y.T, 0.56, solution.t)