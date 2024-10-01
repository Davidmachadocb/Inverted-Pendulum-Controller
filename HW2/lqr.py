import numpy as np
from scipy import integrate
from visual import animate_cart_pendulum, plot_results

# Physical constants
G = 9.81          # Gravity (m/s^2)
L = 0.56          # Length of the shaft (m)
m = 0.356         # Mass of the point (kg)
M = 4.8           # Mass of the cart
V = 0.035         # Viscous friction of the joint

