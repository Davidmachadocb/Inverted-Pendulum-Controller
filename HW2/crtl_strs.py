import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from math import sin

# Physical constants
G = 9.81          # Gravity (m/s^2)
L = 0.56          # Length of the shaft (m)
m = 0.356         # Mass of the point (kg)
M = 4.8           # Mass of the cart
V = 0.035         # Viscous friction of the joint

interator = 0

class FuzzyPDPendulum:
    def __init__(self):
        error = ctrl.Antecedent(np.arange(-np.pi, np.pi, 0.01), 'error')
        delta_error = ctrl.Antecedent(np.arange(-10, 10, 0.01), 'delta_error')
        kp = ctrl.Consequent(np.arange(0, 100, 1), 'kp')
        kd = ctrl.Consequent(np.arange(0, 50, 1), 'kd')

        error['negative'] = fuzz.trimf(error.universe, [-np.pi, -np.pi, 0])
        error['zero'] = fuzz.trimf(error.universe, [-np.pi, 0, np.pi])
        error['positive'] = fuzz.trimf(error.universe, [0, np.pi, np.pi])

        delta_error['negative'] = fuzz.trimf(delta_error.universe, [-10, -10, 0])
        delta_error['zero'] = fuzz.trimf(delta_error.universe, [-10, 0, 10])
        delta_error['positive'] = fuzz.trimf(delta_error.universe, [0, 10, 10])

        kp['low'] = fuzz.trimf(kp.universe, [0, 0, 50])
        kp['medium'] = fuzz.trimf(kp.universe, [0, 50, 100])
        kp['high'] = fuzz.trimf(kp.universe, [50, 100, 100])

        kd['low'] = fuzz.trimf(kd.universe, [0, 0, 25])
        kd['medium'] = fuzz.trimf(kd.universe, [0, 25, 50])
        kd['high'] = fuzz.trimf(kd.universe, [25, 50, 50])

        rules = [
            ctrl.Rule(error['negative'] & delta_error['negative'], (kp['medium'], kd['medium'])),
            ctrl.Rule(error['negative'] & delta_error['zero'], (kp['high'], kd['high'])),
            ctrl.Rule(error['negative'] & delta_error['positive'], (kp['medium'], kd['high'])),
            ctrl.Rule(error['zero'] & delta_error['negative'], (kp['medium'], kd['medium'])),
            ctrl.Rule(error['zero'] & delta_error['zero'], (kp['high'], kd['medium'])),
            ctrl.Rule(error['zero'] & delta_error['positive'], (kp['medium'], kd['medium'])),
            ctrl.Rule(error['positive'] & delta_error['negative'], (kp['medium'], kd['medium'])),
            ctrl.Rule(error['positive'] & delta_error['zero'], (kp['high'], kd['high'])),
            ctrl.Rule(error['positive'] & delta_error['positive'], (kp['medium'], kd['high'])),
        ]


        kp_ctrl = ctrl.ControlSystem(rules)
        self.kp_sim = ctrl.ControlSystemSimulation(kp_ctrl)

    def compute_gains(self, error, delta_error):
        self.kp_sim.input['error'] = error
        self.kp_sim.input['delta_error'] = delta_error
        self.kp_sim.compute()
        kp = self.kp_sim.output['kp']
        kd = self.kp_sim.output['kd']
        return kp, kd

class FuzzyPDCart:
    def __init__(self):
        cart_error = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'cart_error')
        cart_delta_error = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'cart_delta_error')
        kp = ctrl.Consequent(np.arange(0, 100, 1), 'kp')
        kd = ctrl.Consequent(np.arange(0, 50, 1), 'kd')

        cart_error['negative'] = fuzz.trimf(cart_error.universe, [-1, -1, 0])
        cart_error['zero'] = fuzz.trimf(cart_error.universe, [-1, 0, 1])
        cart_error['positive'] = fuzz.trimf(cart_error.universe, [0, 1, 1])

        cart_delta_error['negative'] = fuzz.trimf(cart_delta_error.universe, [-1, -1, 0])
        cart_delta_error['zero'] = fuzz.trimf(cart_delta_error.universe, [-1, 0, 1])
        cart_delta_error['positive'] = fuzz.trimf(cart_delta_error.universe, [0, 1, 1])

        kp['low'] = fuzz.trimf(kp.universe, [0, 0, 50])
        kp['medium'] = fuzz.trimf(kp.universe, [0, 50, 100])
        kp['high'] = fuzz.trimf(kp.universe, [50, 100, 100])

        kd['low'] = fuzz.trimf(kd.universe, [0, 0, 25])
        kd['medium'] = fuzz.trimf(kd.universe, [0, 25, 50])
        kd['high'] = fuzz.trimf(kd.universe, [25, 50, 50])

        rules = [
            ctrl.Rule(cart_error['negative'] & cart_delta_error['negative'], (kp['low'], kd['low'])),
            ctrl.Rule(cart_error['negative'] & cart_delta_error['zero'], (kp['low'], kd['medium'])),
            ctrl.Rule(cart_error['negative'] & cart_delta_error['positive'], (kp['low'], kd['medium'])),
            ctrl.Rule(cart_error['zero'] & cart_delta_error['negative'], (kp['low'], kd['high'])),
            ctrl.Rule(cart_error['zero'] & cart_delta_error['zero'], (kp['low'], kd['low'])),
            ctrl.Rule(cart_error['zero'] & cart_delta_error['positive'], (kp['low'], kd['high'])),
            ctrl.Rule(cart_error['positive'] & cart_delta_error['negative'], (kp['low'], kd['medium'])),
            ctrl.Rule(cart_error['positive'] & cart_delta_error['zero'], (kp['low'], kd['medium'])),
            ctrl.Rule(cart_error['positive'] & cart_delta_error['positive'], (kp['low'], kd['low'])),
        ]

        kp_ctrl = ctrl.ControlSystem(rules)
        self.kp_sim = ctrl.ControlSystemSimulation(kp_ctrl)

    def compute_gains(self, cart_error, cart_delta_error):
        self.kp_sim.input['cart_error'] = cart_error
        self.kp_sim.input['cart_delta_error'] = cart_delta_error
        self.kp_sim.compute()
        kp = self.kp_sim.output['kp']
        kd = self.kp_sim.output['kd']
        return kp, kd

def fuzzy_pd(state, t, fuzzy_controller_pendulum, fuzzy_controller_cart):
    th, th_d, x, x_d = state
    error = th
    delta_error = th_d
    cart_error = x
    cart_delta_error = x_d

    kp_pend, kd_pend = fuzzy_controller_pendulum.compute_gains(error, delta_error)
    kp_cart, kd_cart = fuzzy_controller_cart.compute_gains(cart_error, cart_delta_error)

    u_pendulum = kp_pend * error + kd_pend * delta_error
    setpoint = setpoint_function(t)
    u_cart = kp_cart * (cart_error - setpoint) + kd_cart * cart_delta_error

    u = u_pendulum + u_cart + 0.05 * (np.sin(t) * np.exp(-np.cos(t)) - t // 4 - np.exp(-t))

    damping_term = V * th_d / (M * L)
    ds = np.zeros_like(state)
    ds[0] = th_d
    ds[1] = (G * th - u - damping_term) / L
    ds[2] = x_d
    ds[3] = u

    return ds

def compute_control_action_fuzzy_pd(state, t, fuzzy_controller_pendulum, fuzzy_controller_cart):
    th, th_d, x, x_d = state
    error = th
    delta_error = th_d
    cart_error = x
    cart_delta_error = x_d

    kp_pend, kd_pend = fuzzy_controller_pendulum.compute_gains(error, delta_error)
    kp_cart, kd_cart = fuzzy_controller_cart.compute_gains(cart_error, cart_delta_error)

    u_pendulum = kp_pend * error + kd_pend * delta_error
    setpoint = setpoint_function(t)
    u_cart = kp_cart * (cart_error - setpoint) + kd_cart * cart_delta_error

    u = u_pendulum + u_cart + 0.05 * (np.sin(t) * np.exp(-np.cos(t)) - t // 4 - np.exp(-t))
    return u

    
def setpoint_function(t):
    return (t // 6)

def cascade_pd(state, t, controller_constants):
    K_p, K_d, Kx_p, Kx_d = controller_constants
    ds = np.zeros_like(state)
    _th, _th_d, _x, _x_d = state

    damping_term = V * _th_d / (M * L)

    setpoint = setpoint_function(t)
    # Outer loop control (cart position)
    u1 = Kx_p * (_x - setpoint) + Kx_d * _x_d

    # Calculate the derivative of u1
    u1_d = Kx_p * _x_d

    # Inner loop control (pendulum angle)
    u2 = K_p * (_th - u1) + K_d * (_th_d - u1_d) + \
        0.05 * (np.sin(t) * np.exp(-np.cos(t)) - t // 4 - np.exp(-t))

    ds[0] = _th_d
    ds[1] = (G * _th - u2 - damping_term) / L
    ds[2] = _x_d
    ds[3] = _th

    return ds

def compute_control_action_cascade_pd(state, t, controller_constants):
    K_p, K_d, Kx_p, Kx_d = controller_constants
    _th, _th_d, _x, _x_d = state

    setpoint = setpoint_function(t)
    u1 = Kx_p * (_x - setpoint) + Kx_d * _x_d
    u1_d = Kx_p * _x_d

    u2 = K_p * (_th - u1) + K_d * (_th_d - u1_d) + \
        0.05 * (np.sin(t) * np.exp(-np.cos(t)) - t // 4 - np.exp(-t))
    return u2


def normal_pd(state, t, controller_constants):
    K_p, K_d, Kx_p, Kx_d = controller_constants
    ds = np.zeros_like(state)
    _th, _th_d, _x, _x_d = state

    damping_term = V * _th_d / (m * L)

    setpoint = setpoint_function(t=t)

    u = K_p * (_th) + K_d * _th_d + Kx_p * (_x - setpoint) + Kx_d * _x_d + \
        0.5 * (np.sin(t) * np.exp(-np.cos(t)) - t // 4 - np.exp(-t))

    ds[0] = _th_d
    ds[1] = (G * _th - u - damping_term) / L
    ds[2] = _x_d
    ds[3] = u / (m + M)
    return ds

def compute_control_action_normal_pd(state, t, controller_constants):
    K_p, K_d, Kx_p, Kx_d = controller_constants
    _th, _th_d, _x, _x_d = state

    setpoint = setpoint_function(t=t)

    u = K_p * (_th) + K_d * _th_d + Kx_p * (_x - setpoint) + Kx_d * _x_d + \
        0.5 * (np.sin(t) * np.exp(-np.cos(t)) - t // 4 - np.exp(-t))
    return u
