import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

G = 9.81          # Gravity (m/s^2)
L = 0.56          # Length of the shaft (m)
M = 0.356         # Mass of the point (kg)
V = 0.035         # Viscous friction of the joint

# Initialize the state and time array
th = np.pi / 3  # Pendulum angle (rad)
th_d = 0  # Pendulum angular velocity (rad/s)
x = 0.0  # Cart position (m)
x_d = 0  # Cart velocity (m/s)
s0 = np.array([th, th_d, x, x_d])

dt = 0.05
Tmax = 5
t = np.arange(0.0, Tmax, dt)

print(len(t))

# Normal PD control
normal_constants = [54.4, 8, 8, 8]  # for normal
control_actions = []  # Global list to store control actions
actual_times = []  # Global list to store the actual time points

def normal_pd(state, t, controller_constants):
    global control_actions, actual_times
    K_p, K_d, Kx_p, Kx_d = controller_constants
    ds = np.zeros_like(state)
    _th, _th_d, _x, _x_d = state

    damping_term = V * _th_d / (M * L)
    u = K_p * (_th) + K_d * _th_d + Kx_p * _x + Kx_d * _x_d

    ds[0] = _th_d
    ds[1] = (G * _th - u - damping_term) / L
    ds[2] = _x_d
    ds[3] = u

    control_actions.append(u)
    return ds

# Perform the integration
solution_normal = integrate.odeint(func=normal_pd, y0=s0, t=t, args=(normal_constants,), atol=1e-8, rtol=1e-6)

# Interpolate the control actions to match the time array t
control_actions = np.array(control_actions)
actual_times = np.array(actual_times)

print(max(control_actions))