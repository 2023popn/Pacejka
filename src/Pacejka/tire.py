import numpy as np

class Tire:
    def __init__(self, name, mu_peak, cornering_stiffness_per_deg, load_sensitivity_n, f_z0,  c=1.4, e=-0.1):
        self.name = name
        self.mu_peak = mu_peak
        self.cornering_stiffness_per_deg = cornering_stiffness_per_deg
        self.load_sensitivity_n = load_sensitivity_n
        self.f_z0 = f_z0

        self.params = {
            'x': {'B': 10.0, 'C': 1.6, 'D' : 0, 'E': 0.1},  # Longitudinal
            'y': {'B': 8.0, 'C': 1.4, 'D' : 0, 'E': -0.1},  # Lateral
            'z': {'B': 9.0, 'C': 2.0, 'D' : 0, 'E': 0.5}  # Aligning Moment
        }

        self.c = c
        self.e = e

        self.b = 0
        self.d = 0

    def calculate_coefficients(self, f_z):
        mu_effective = self.mu_peak * pow(f_z / self.f_z0, self.load_sensitivity_n - 1)
        self.d = mu_effective * f_z

        c_alpha_current_deg = self.cornering_stiffness_per_deg * ((f_z / self.f_z0) ** self.load_sensitivity_n)
        c_alpha_rad = c_alpha_current_deg * (180.0 / np.pi)

        self.b = c_alpha_rad / (self.c * self.d) if self.d > 0 else 0

    def
