import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the FuzzyPDPendulum class
class FuzzyPDPendulum:
    def __init__(self):
        # Define Antecedents
        self.error = ctrl.Antecedent(np.arange(-np.pi, np.pi, 0.01), 'error')
        self.delta_error = ctrl.Antecedent(np.arange(-10, 10, 0.01), 'delta_error')
        # Define Consequents
        self.kp = ctrl.Consequent(np.arange(0, 100, 1), 'kp')
        self.kd = ctrl.Consequent(np.arange(0, 50, 1), 'kd')

        # Define membership functions for error
        self.error['negative'] = fuzz.trimf(self.error.universe, [-np.pi, -np.pi, 0])
        self.error['zero'] = fuzz.trimf(self.error.universe, [-np.pi, 0, np.pi])
        self.error['positive'] = fuzz.trimf(self.error.universe, [0, np.pi, np.pi])

        # Define membership functions for delta_error
        self.delta_error['negative'] = fuzz.trimf(self.delta_error.universe, [-10, -10, 0])
        self.delta_error['zero'] = fuzz.trimf(self.delta_error.universe, [-10, 0, 10])
        self.delta_error['positive'] = fuzz.trimf(self.delta_error.universe, [0, 10, 10])

        # Define membership functions for kp
        self.kp['low'] = fuzz.trimf(self.kp.universe, [0, 0, 50])
        self.kp['medium'] = fuzz.trimf(self.kp.universe, [0, 50, 100])
        self.kp['high'] = fuzz.trimf(self.kp.universe, [50, 100, 100])

        # Define membership functions for kd
        self.kd['low'] = fuzz.trimf(self.kd.universe, [0, 0, 25])
        self.kd['medium'] = fuzz.trimf(self.kd.universe, [0, 25, 50])
        self.kd['high'] = fuzz.trimf(self.kd.universe, [25, 50, 50])

        # Define rules
        self.rules = [
            ctrl.Rule(self.error['negative'] & self.delta_error['negative'], (self.kp['medium'], self.kd['medium'])),
            ctrl.Rule(self.error['negative'] & self.delta_error['zero'], (self.kp['high'], self.kd['high'])),
            ctrl.Rule(self.error['negative'] & self.delta_error['positive'], (self.kp['medium'], self.kd['high'])),
            ctrl.Rule(self.error['zero'] & self.delta_error['negative'], (self.kp['medium'], self.kd['medium'])),
            ctrl.Rule(self.error['zero'] & self.delta_error['zero'], (self.kp['high'], self.kd['medium'])),
            ctrl.Rule(self.error['zero'] & self.delta_error['positive'], (self.kp['medium'], self.kd['medium'])),
            ctrl.Rule(self.error['positive'] & self.delta_error['negative'], (self.kp['medium'], self.kd['medium'])),
            ctrl.Rule(self.error['positive'] & self.delta_error['zero'], (self.kp['high'], self.kd['high'])),
            ctrl.Rule(self.error['positive'] & self.delta_error['positive'], (self.kp['medium'], self.kd['high'])),
        ]

        # Create control system and simulation
        self.kp_ctrl = ctrl.ControlSystem(self.rules)
        self.kp_sim = ctrl.ControlSystemSimulation(self.kp_ctrl)

    def compute_gains(self, error, delta_error):
        self.kp_sim.input['error'] = error
        self.kp_sim.input['delta_error'] = delta_error
        self.kp_sim.compute()
        kp = self.kp_sim.output['kp']
        kd = self.kp_sim.output['kd']
        return kp, kd

# Define the FuzzyPDCart class
class FuzzyPDCart:
    def __init__(self):
        # Define Antecedents
        self.cart_error = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'cart_error')
        self.cart_delta_error = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'cart_delta_error')
        # Define Consequents
        self.kp = ctrl.Consequent(np.arange(0, 100, 1), 'kp')
        self.kd = ctrl.Consequent(np.arange(0, 50, 1), 'kd')

        # Define membership functions for cart_error
        self.cart_error['negative'] = fuzz.trimf(self.cart_error.universe, [-1, -1, 0])
        self.cart_error['zero'] = fuzz.trimf(self.cart_error.universe, [-1, 0, 1])
        self.cart_error['positive'] = fuzz.trimf(self.cart_error.universe, [0, 1, 1])

        # Define membership functions for cart_delta_error
        self.cart_delta_error['negative'] = fuzz.trimf(self.cart_delta_error.universe, [-1, -1, 0])
        self.cart_delta_error['zero'] = fuzz.trimf(self.cart_delta_error.universe, [-1, 0, 1])
        self.cart_delta_error['positive'] = fuzz.trimf(self.cart_delta_error.universe, [0, 1, 1])

        # Define membership functions for kp
        self.kp['low'] = fuzz.trimf(self.kp.universe, [0, 0, 50])
        self.kp['medium'] = fuzz.trimf(self.kp.universe, [0, 50, 100])
        self.kp['high'] = fuzz.trimf(self.kp.universe, [50, 100, 100])

        # Define membership functions for kd
        self.kd['low'] = fuzz.trimf(self.kd.universe, [0, 0, 25])
        self.kd['medium'] = fuzz.trimf(self.kd.universe, [0, 25, 50])
        self.kd['high'] = fuzz.trimf(self.kd.universe, [25, 50, 50])

        # Define rules
        self.rules = [
            ctrl.Rule(self.cart_error['negative'] & self.cart_delta_error['negative'], (self.kp['low'], self.kd['low'])),
            ctrl.Rule(self.cart_error['negative'] & self.cart_delta_error['zero'], (self.kp['low'], self.kd['medium'])),
            ctrl.Rule(self.cart_error['negative'] & self.cart_delta_error['positive'], (self.kp['low'], self.kd['medium'])),
            ctrl.Rule(self.cart_error['zero'] & self.cart_delta_error['negative'], (self.kp['low'], self.kd['high'])),
            ctrl.Rule(self.cart_error['zero'] & self.cart_delta_error['zero'], (self.kp['low'], self.kd['low'])),
            ctrl.Rule(self.cart_error['zero'] & self.cart_delta_error['positive'], (self.kp['low'], self.kd['high'])),
            ctrl.Rule(self.cart_error['positive'] & self.cart_delta_error['negative'], (self.kp['low'], self.kd['medium'])),
            ctrl.Rule(self.cart_error['positive'] & self.cart_delta_error['zero'], (self.kp['low'], self.kd['medium'])),
            ctrl.Rule(self.cart_error['positive'] & self.cart_delta_error['positive'], (self.kp['low'], self.kd['low'])),
        ]

        # Create control system and simulation
        self.kp_ctrl = ctrl.ControlSystem(self.rules)
        self.kp_sim = ctrl.ControlSystemSimulation(self.kp_ctrl)

    def compute_gains(self, cart_error, cart_delta_error):
        self.kp_sim.input['cart_error'] = cart_error
        self.kp_sim.input['cart_delta_error'] = cart_delta_error
        self.kp_sim.compute()
        kp = self.kp_sim.output['kp']
        kd = self.kp_sim.output['kd']
        return kp, kd

# Modified function to plot membership functions in a grid layout
def plot_membership_functions(controller, controller_name):
    # Collect antecedents and consequents
    if hasattr(controller, 'error'):
        variables = [controller.error, controller.delta_error, controller.kp, controller.kd]
    else:
        variables = [controller.cart_error, controller.cart_delta_error, controller.kp, controller.kd]
    
    num_vars = len(variables)
    cols = 2
    rows = num_vars // cols + (num_vars % cols > 0)
    
    # Create subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        for term_name, mf in var.terms.items():
            ax.plot(var.universe, mf.mf, label=term_name)
        ax.set_title(f"{controller_name} - {var.label}")
        ax.legend()
    
    # Hide any unused subplots
    for idx in range(len(variables), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

# Function to plot rule surfaces
def plot_rule_surface(controller, controller_name):
    # Determine input ranges and labels
    if hasattr(controller, 'error'):
        x_range = controller.error.universe
        y_range = controller.delta_error.universe
        input1_label = 'Error'
        input2_label = 'Delta Error'
    else:
        x_range = controller.cart_error.universe
        y_range = controller.cart_delta_error.universe
        input1_label = 'Cart Error'
        input2_label = 'Cart Delta Error'

    # Reduce the number of points for plotting to speed up the computation
    x_samples = np.linspace(x_range.min(), x_range.max(), 50)
    y_samples = np.linspace(y_range.min(), y_range.max(), 50)
    x, y = np.meshgrid(x_samples, y_samples)
    z_kp = np.zeros_like(x)
    z_kd = np.zeros_like(x)

    # Compute the output for each pair of inputs
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if hasattr(controller, 'error'):
                controller.kp_sim.input['error'] = x[i, j]
                controller.kp_sim.input['delta_error'] = y[i, j]
            else:
                controller.kp_sim.input['cart_error'] = x[i, j]
                controller.kp_sim.input['cart_delta_error'] = y[i, j]
            try:
                controller.kp_sim.compute()
                z_kp[i, j] = controller.kp_sim.output['kp']
                z_kd[i, j] = controller.kp_sim.output['kd']
            except:
                z_kp[i, j] = np.nan
                z_kd[i, j] = np.nan

    # Plot Kp surface
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x, y, z_kp, cmap='viridis')
    ax1.set_title(f'{controller_name} - Kp Surface')
    ax1.set_xlabel(input1_label)
    ax1.set_ylabel(input2_label)
    ax1.set_zlabel('Kp')

    # Plot Kd surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, z_kd, cmap='plasma')
    ax2.set_title(f'{controller_name} - Kd Surface')
    ax2.set_xlabel(input1_label)
    ax2.set_ylabel(input2_label)
    ax2.set_zlabel('Kd')

    plt.tight_layout()
    plt.show()

# Function to plot all fuzzy plots for a controller
def plot_fuzzy_controller(controller, controller_name):
    print(f"Plotting membership functions for {controller_name}...")
    plot_membership_functions(controller, controller_name)
    print(f"Plotting rule surfaces for {controller_name}...")
    plot_rule_surface(controller, controller_name)

# Example usage
if __name__ == "__main__":
    # Initialize controllers
    pendulum_controller = FuzzyPDPendulum()
    cart_controller = FuzzyPDCart()

    # Plot fuzzy plots for Pendulum Controller
    plot_fuzzy_controller(pendulum_controller, 'Pendulum Controller')

    # Plot fuzzy plots for Cart Controller
    plot_fuzzy_controller(cart_controller, 'Cart Controller')