# Inverted-Pendulum-Controller

This project contains simulation and control algorithms for an inverted pendulum on a cart system. The inverted pendulum is a classic control problem used to test and implement control strategies for inherently unstable systems.

## Overview
The system models and controls the dynamics of an inverted pendulum on a cart. Several control strategies are implemented and compared, including:

- **PD Controller**: A basic proportional-derivative controller for balancing the pendulum.
- **Cascade PD Controller**: An improvement over the basic PD, with nested loops for controlling both the cart and pendulum.
- **Fuzzy PD Controller**: A flexible controller using fuzzy logic to handle non-linearity and uncertainty in the system.
- **Linear Quadratic Regulator (LQR)**: An optimal control strategy minimizing the system's energy and control effort.
- **Model Predictive Control (MPC)**: A computationally expensive but highly effective control strategy for minimizing control effort and maintaining system stability over a prediction horizon.

## Features
- **State-space modeling**: Uses Lagrangian mechanics to derive the state-space representation of the system.
- **Linearization**: System is linearized around the upright equilibrium for simplified analysis and control design.
- **Stability Analysis**: Includes BIBO and Lyapunov's stability analysis, along with controllability and observability assessments.
- **Controller Comparisons**: Performance of PD, cascade PD, fuzzy PD, LQR, and MPC controllers is compared in terms of response time, stability, and energy efficiency.

## Simulations
Simulations are provided for both nonlinear and linearized models, showing the system's behavior under different controllers. Python is used to simulate and animate the system.

## Conclusion
MPC performed the best in terms of tracking accuracy and minimal control effort, though it requires significant computational resources. LQR is a strong alternative with reliable performance. PD-based controllers (standard and fuzzy) provide simpler implementations but struggle with efficiency and stability.
