import numpy as np
import cvxpy as cp
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import control

# Define system constants
M = 1.0      # Mass of the cart (kg)
L = 0.56     # Length of the pendulum (m)
G = 9.81     # Acceleration due to gravity (m/s^2)
D_pend = 0.1 # Damping coefficient for pendulum
m = 0.356    # Mass of the pendulum bob (kg)

dt = 0.05    # Time step

A = np.array([
    [0, 1, 0, 0],
    [-G / L, -D_pend / L, 0, 0],
    [0, 0, 0, 1],
    [-m * G / (m + M), -m * D_pend / (m + M), 0, 0]
])

B = np.array([
    [0],
    [-1 / L],
    [0],
    [1 / (m + M)]
])

C = np.eye(4)
D_matrix = np.zeros((4, 1))

# Discretize the system
sys = control.StateSpace(A, B, C, D_matrix)
sys_discrete = control.c2d(sys, dt, method='zoh')

A_zoh = np.array(sys_discrete.A)
B_zoh = np.array(sys_discrete.B)

def run_mpc():
    cost = 0.
    constr = [x[:, 0] == x_init]
    for t_step in range(N):
        # Cost function
        cost += cp.quad_form(x[:, t_step] - xr[:, t_step], Q) + cp.quad_form(u[:, t_step], R)
        # Control input constraints
        constr += [cp.norm(u[:, t_step], 'inf') <= 10.]
        # System dynamics constraints
        constr += [x[:, t_step + 1] == A_zoh @ x[:, t_step] + B_zoh @ u[:, t_step]]
    # Terminal cost
    cost += cp.quad_form(x[:, N] - xr[:, N], Q)
    # Optimization problem
    problem = cp.Problem(cp.Minimize(cost), constr)
    return problem

[nx, nu] = B_zoh.shape

Q = sparse.diags([100., 1., 100., 1.])
R = np.array([[1]])                     

x0 = np.array([np.pi / 4, 0., 0., 0.])  # Initial state

# MPC horizon length
N = 20

x = cp.Variable((nx, N+1))
u = cp.Variable((nu, N))
x_init = cp.Parameter(nx)
xr = cp.Parameter((nx, N+1))

nsim = 500 
t = np.arange(0, nsim * dt + dt, dt)

x_desired_full = (t // 6)  # Converts floor(t/6) to setpoint increments

solution = [x0.copy()]

ctrl_effort = []

# Simulation loop
for i in range(nsim):
    prob = run_mpc()
    x_init.value = x0

    x_desired_horizon = x_desired_full[i:i+N+1]
    if len(x_desired_horizon) < N+1:
        x_desired_horizon = np.pad(x_desired_horizon, (0, N+1 - len(x_desired_horizon)), 'edge')

    xr_value = np.zeros((nx, N+1))
    xr_value[0, :] = 0.0                   # Desired theta
    xr_value[1, :] = 0.0                   # Desired theta_dot
    xr_value[2, :] = 0     # Desired x
    xr_value[3, :] = 0.0                   # Desired x_dot
    xr.value = xr_value

    # Solve the MPC optimization problem
    prob.solve(solver=cp.OSQP, warm_start=True)

    # Retrieve the first control input
    u_k = u[:, 0].value
    if u_k is None:
        u_k = np.array([0.0])  # Handle None values if solver fails
    # Update the state using the system dynamics
    x0 = A_zoh @ x0 + B_zoh @ u_k

    # Append the new state to the solution list
    solution.append(x0.copy())

    # Store the control effort
    ctrl_effort.append(u_k.item())

    # Optional: Print progress every 50 steps
    if (i+1) % 50 == 0:
        print(f'TIME: {round((i+1)*dt, 2)} s, STATES: {np.round(x0, 2)}')

solution = np.array(solution)

def plot_results(t, solution, control_actions=None):
    plt.figure(figsize=(12, 10))

    # Pendulum Angle
    plt.subplot(3, 1, 1)
    plt.plot(t[:len(solution)], solution[:, 0], label='Pendulum Angle (θ)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Pendulum Angle Over Time')
    plt.legend()
    plt.grid(True)

    # Cart Position and Setpoint
    plt.subplot(3, 1, 2)
    plt.plot(t[:len(solution)], solution[:, 2], label='Cart Position (x)', color='orange')
#    plt.plot(t[:len(solution)], x_desired_full[:len(solution)], label='Cart Setpoint', color='green', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Cart Position Over Time')
    plt.legend()
    plt.grid(True)

    # Control Action
    if control_actions is not None:
        plt.subplot(3, 1, 3)
        plt.plot(t[:len(control_actions)], control_actions, label='Control Action (u)', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input (u)')
        plt.title('Control Action Over Time')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Define the animation function
def animate_cart_pendulum(solution, L, t, cart_width=0.3, cart_height=0.2, interval=25):
    ths  = solution[:, 0]
    xs   = solution[:, 2]

    pxs = L * np.sin(ths) + xs
    pys = L * np.cos(ths)

    fig, ax = plt.subplots()
    ax.set_xlim(np.min(xs) - 1, np.max(xs) + 1)
    ax.set_ylim(-0.5, L + 1)
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'Time = %.2f s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    cart = Rectangle((0, 0), cart_width, cart_height,
                     linewidth=1, edgecolor='black', facecolor='blue')
    ax.add_patch(cart)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        cart.set_xy((xs[0] - cart_width / 2, 0))
        return line, time_text, cart

    def animate_frame(i):
        if i >= len(t):
            i = len(t) - 1
        thisx = [xs[i], pxs[i]]
        thisy = [cart_height / 2, pys[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i * dt))
        cart.set_x(xs[i] - cart_width / 2)
        return line, time_text, cart

    ani = animation.FuncAnimation(fig, animate_frame, frames=len(t),
                                  interval=interval, blit=True, init_func=init)
    ani.save('catching-pendulum.gif', writer='pillow', fps=20)
    plt.show()
    return ani


plot_results(t, solution, control_actions=ctrl_effort)

animate_cart_pendulum(solution, L, t)