import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from numpy import sin, cos
import numpy as np

# Physical constants
g = 9.81          # gravity
L = 0.56          # Length of the shaft
m = 0.356         # mass of the point
d_mf = 0.035      # Viscous friction of the joint

# Simulation time
dt = 0.05
Tmax = 20
t = np.arange(0.0, Tmax, dt)

# Initial conditions
th = np.pi/(2)  # Pendulum angle
th_d = 0          # Pendulum angular velocity
x = 0.0           # Cart position
x_d = -0.05       # Cart velocity

state = np.array([th, th_d, x, x_d])

def derivatives(state, t):
    ds = np.zeros_like(state)

    _th   = state[0]
    _th_d = state[1]
    _x    = state[2]
    _x_d  = state[3]

    damping_term = d_mf * _th_d / (m * L)

    u =  21.3336153*_th + 4.87773537 * _th_d + 1.16264377*_x + 3.10709072*_x_d

    ds[0] = _th_d
    ds[1] = (g * _th - u - damping_term) / L
    ds[2] = _x_d
    ds[3] = u

    return ds

solution = integrate.odeint(derivatives, state, t)

ths  = solution[:, 0]
Ys   = solution[:, 1]
xs   = solution[:, 2]
x_ds = solution[:, 3]

pxs = L * sin(ths) + xs
pys = L * cos(ths)

fig, ax = plt.subplots()
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'Time = %.1f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# Cart dimensions
cart_width = 0.3
cart_height = 0.2

# Add cart
cart = Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height,
                 linewidth=1, edgecolor='black', facecolor='blue')
ax.add_patch(cart)

def init():
    line.set_data([], [])
    time_text.set_text('')
    cart.set_xy((-cart_width / 2, -cart_height / 2))
    return line, time_text, cart

def animate(i):
    thisx = [xs[i], pxs[i]]
    thisy = [0, pys[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    cart.set_x(xs[i] - cart_width / 2)
    return line, time_text, cart

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=25, blit=True, init_func=init)
plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Sergey Royz'), bitrate=1800)
ani.save('catching-pendulum.mp4', writer=writer)