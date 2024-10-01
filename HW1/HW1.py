"""
----------------Homework 1----------------
|Name: David Machado Couto Bezerra       |
|matrícula: 475664                       |
|Curso: Engenharia de Computação         |
------------------------------------------
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
Question 1:
    Equations:
        th'' = (g * sin(th) - u * cos(th) - d_mf * th'/(m * L)) / L,
        u = input control
    States:
        x = [th th']
        input = u
        output = th
    Objectives:
        Simulation of pendulum and stabilizing with a controller with an angle of th = 0.
"""

def inverted_pendulum(state, t, step_time, step_value):
    # Constants
    g = 9.81      # gravity
    L = 0.56      # Length of the shaft
    m = 0.356     # mass of the point
    d_mf = 0.035  # Viscous friction of the joint

    theta, theta_dot = state
    u = step_value if t >= step_time else 0
    
    damping_term = d_mf * theta_dot / (m * L)
    theta_ddot = (g * np.sin(theta) - u * np.cos(theta) - damping_term) / L
    
    return [theta_dot, theta_ddot]

# Initial conditions
x0 = [np.pi / 4, 0]

step_time = 0.5
step_value = -1  

t = np.linspace(0, 5*np.pi, 1000)
x = odeint(inverted_pendulum, x0, t, args=(step_time, step_value))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

ax1.axhline(y=np.pi, color='g', linestyle='--', label=r'$\pi$')
ax1.plot(t, x[:, 0], 'b-', label=r'$\theta(t)$')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel(r'$\theta(t)$')
ax1.set_title('Angle Response')
ax1.set_ylim(0, 6)
ax1.legend()
ax1.grid(True)

ax2.axhline(y=0, color='g', linestyle='--', label=r'$0$')
ax2.plot(t, x[:, 1], 'r-', label=r'$\dot{\theta}(t)$')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel(r'$\dot{\theta}(t)$')
ax2.set_title('Angular Velocity Response')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

"""
Question 2:
    Linearisation:
        Steady state point: [0 0]
        Taylor series around ss point: 
            cos(th) ~ 1
            sin(th) ~ 0
    State space model:
        x' = Ax + Bu
        y  = Cx + Du

        A = [[0, 1]
            [g/L, -d_mf/(m*L)]]

        B = [[0]
            [-1/L]]

        C = [[1]
            [0]]
        
        D = [0]
"""

import control as ct

g = 9.81      # gravity
L = 0.56      # Length of the shart
m = 0.356     # mass of the point
d_mf = 0.035  # Viscous friction of the joint

A=np.array([[0, 1], [g/L, -d_mf/(m*L)]])
B=np.array([[0], [-1/L]])
C=np.array([[1, 0]])
D=np.array([[0]])

linear_ip = ct.ss(A, B, C, D)

u = np.array([step_value if time >= step_time else 0 for time in t])

T, x_linear = ct.forced_response(linear_ip, T=t, U=u, X0=x0)

plt.axhline(y=np.pi, color='g', linestyle='--', label=r'$\pi$')
plt.plot(T, x_linear, 'b-', label=r'$\theta_L(t)$')
plt.plot(T, x[:, 0], 'r-', label=r'$\theta(t)$')
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta(t)$')
plt.title('Angle Response')
plt.ylim(-6, 6)
plt.xlim(0, 2.5)
plt.legend()
plt.grid(True)
plt.show()

"""
Question 3:
    Transfer function: 
        H(s) = (-1/L)/(s^2+(d_mf/mL^2)s -g/L)
"""

tf_ip = ct.ss2tf(linear_ip)
print("Transfer fuction of the Inverted Pendulum:", tf_ip)
T, step_response = ct.forced_response(tf_ip, T=t, U=u)

plt.axhline(y=np.pi, color='g', linestyle='--', label=r'$\pi$')
plt.plot(T, step_response, 'b-', label=r'$\theta_L(t)$')
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta(t)$')
plt.title('Angle Response')
plt.ylim(-6, 6)
plt.xlim(0, 2.5)
plt.legend()
plt.grid(True)
plt.show()

info = ct.step_info(x_linear, T)
print("\n Info: \n", info)

"""
Question 4:
    Poles:
        s_1 = 4.0988
        s_2 = -4.2744
    Lyapunov Function:
        E(th, th') = 1/2 * (ml^2 (th')^2) + mglcos(th)
        E' = th' (cos(th))
"""
from numpy.linalg import eig

#poles
plt.figure()
ct.pzmap(tf_ip, plot=True, title='Pole-Zero Map')
plt.show()

#eigenvalues
w, v = eig(A)
print("\n\nEigenvalues of A: ", w)

#Lyapunov linear
Q = np.array([[1, 0], [0, 1]])
P = ct.lyap(A, Q)
print("\n\nP MATRIX: ", P)
w, v = eig(P)
print("\n\nEigenvalues of P: ", w)

def Energy_derivative(X, Y):
    return -Y * np.cos(X)

#Lyapunov derivative
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = Energy_derivative(X, Y)
z_plane = np.zeros(X.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, z_plane, color='g', alpha=0.5, label=r'$z=0$')
ax.axhline(y=np.pi, color='g', linestyle='--', label=r'$\pi$')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')
ax.set_zlabel(r'$\dot{E}$')
plt.show()

from numpy.linalg import matrix_rank

#Controllability
controllability_matrix = ct.ctrb(A, B)
is_controllable = matrix_rank(controllability_matrix) == len(A)
print(f"Controllability: {'YES' if is_controllable else 'NO'} (Rank: {matrix_rank(controllability_matrix)})")

#Observability
observability_matrix = ct.obsv(A, C)
is_observable = matrix_rank(observability_matrix) == len(A)
print(f"Observability: {'YES' if is_observable else 'NO'} (Rank: {matrix_rank(observability_matrix)})")