import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

def animate_cart_pendulum(solution, L, t, cart_width=0.3, cart_height=0.2, interval=25):
    ths  = solution[:, 0]
    xs   = solution[:, 2]

    pxs = L * np.sin(ths) + xs
    pys = L * np.cos(ths)

    fig, ax = plt.subplots()
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'Time = %.1f s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

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
        time_text.set_text(time_template % (i * (t[1] - t[0])))
        cart.set_x(xs[i] - cart_width / 2)
        return line, time_text, cart

    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=interval, blit=True, init_func=init)
    ani.save('catching-pendulum.gif', writer='pillow', fps=20)
    plt.show()
    return ani

def plot_results(t, solution, control_actions=None):
    plt.figure(figsize=(10, 8))

    # Pendulum angle
    plt.subplot(3, 1, 1)
    plt.plot(t, solution[:, 0], label='Pendulum Angle (θ)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Pendulum Angle Over Time')
    plt.legend()
    plt.grid()

    # Cart position
    plt.subplot(3, 1, 2)
    plt.plot(t, solution[:, 2], label='Cart Position (x)', color='orange')
    plt.plot(t, (t//6), label='Cart Setpoint', color='blue', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Cart Position Over Time')
    plt.legend()
    plt.grid()

    # Control action
    if control_actions is not None:
        plt.subplot(3, 1, 3)
        plt.plot(t, control_actions, label='Control Action (u)', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Action Over Time')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()
