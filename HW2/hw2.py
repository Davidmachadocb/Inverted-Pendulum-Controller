import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np
from visual import plot_results, animate_cart_pendulum
from crtl_strs import (
    fuzzy_pd,
    compute_control_action_fuzzy_pd,
    FuzzyPDPendulum,
    FuzzyPDCart,
    setpoint_function
)

def main():
    # Initialize the state and time array
    th = np.pi / 3    # Pendulum angle (rad)
    th_d = 0          # Pendulum angular velocity (rad/s)
    x = 0.0           # Cart position (m)
    x_d = 0           # Cart velocity (m/s)
    s0 = np.array([th, th_d, x, x_d])

    dt = 0.05
    Tmax = 25
    t = np.arange(0.0, Tmax, dt)

    # Fuzzy PD control
    fuzzy_controller_pendulum = FuzzyPDPendulum()
    fuzzy_controller_cart = FuzzyPDCart()
    solution_fuzzy = integrate.odeint(
        func=fuzzy_pd,
        y0=s0,
        t=t,
        args=(fuzzy_controller_pendulum, fuzzy_controller_cart),
        atol=1e-8,
        rtol=1e-6
    )

    # Compute control actions for fuzzy control
    control_actions_fuzzy = []
    for i in range(len(t)):
        state = solution_fuzzy[i, :]
        current_time = t[i]
        u = compute_control_action_fuzzy_pd(state, current_time, fuzzy_controller_pendulum, fuzzy_controller_cart)
        control_actions_fuzzy.append(u)

    # Plot results
    plot_results(t, solution_fuzzy, control_actions=control_actions_fuzzy)
    #animate_cart_pendulum(solution_fuzzy, 0.56, t)

if __name__ == "__main__":
    main()