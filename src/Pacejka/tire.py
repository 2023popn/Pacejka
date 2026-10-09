import numpy as np

class Tire:
    """
    Class that holds tire parameters and determines forces based on applied conditions

    
    """

    def __init__(self, name, mu_peak, cornering_stiffness_per_deg=None, longitudinal_stiffness=None, load_sensitivity_n=None, f_z0=None, b_x=None, c_x=1.4, e_x=0.1, b_y=None, c_y=1.4, e_y=-0.1, b_constant=False):
        self.name = name
        self.mu_peak = mu_peak
        self.cornering_stiffness_per_deg = cornering_stiffness_per_deg
        self.longitudinal_stiffness = longitudinal_stiffness
        self.load_sensitivity_n = load_sensitivity_n
        self.f_z0 = f_z0

        self.params = {
            'x': {'B': 10.0, 'C': c_x, 'D' : 0, 'E': e_x},  # Longitudinal
            'y': {'B': 8.0, 'C': c_y, 'D' : 0, 'E': e_y},  # Lateral
            # 'z': {'B': 9.0, 'C': 2.0, 'D' : 0, 'E': 0.5}  # Aligning Moment
        }

    @classmethod
    def from_constants(cls, compound_name, mu_base, b_x, c_x, e_x, b_y=None, c_y=None, e_y=None):
        """Initializer for constants only. Assumes constant b"""
        if b_y is None:
            b_y = b_x
        if c_y is None:
            c_y = c_x
        if e_y is None:
            e_y = e_x

        return cls(compound_name, mu_base, b_x=b_x, c_x=c_x, e_x=e_x, b_y=b_y, c_y=c_y, e_y=e_y, b_constant=True)

    @classmethod
    def from_macro_properties(cls, compound_name, mu_base, corner_stiffness_per_deg, longitudinal_stiffness, load_sensitivity_n=0.9, f_z0=150):
        return cls(compound_name, mu_base, cornering_stiffness_per_deg=corner_stiffness_per_deg, longitudinal_stiffness=longitudinal_stiffness, load_sensitivity_n=load_sensitivity_n, f_z0=f_z0)

    def _magic_formula_core(self, x_input, params:dict):
        """
        Magic formula calculation
        """
        b = params['B']
        c = params['C']
        d = params['D']
        e = params['E']

        term = b * x_input
        inner_atan = np.arctan(term)
        argument = c * np.arctan(term - e * (term - inner_atan))
        return d * np.sin(argument)

    def calculate_forces(self, f_z, alpha, kappa, friction_circle=True):
        self.calculate_coefficients(f_z)

        lateral_force = self._magic_formula_core(alpha, self.params['y'])
        longitudinal_force = self._magic_formula_core(kappa, self.params['x'])

        if friction_circle:
            longitudinal_force, lateral_force = self.friction_circle(longitudinal_force, lateral_force, f_z)

        return lateral_force, longitudinal_force

    def calculate_coefficients(self, f_z):
        ## Determine D (peak load) from mu and load
        mu_effective = self.mu_peak * pow(f_z / self.f_z0, self.load_sensitivity_n - 1)
        d = mu_effective * f_z
        self.params['x']['D'] = self.params['y']['D'] = d # Assuming uniform coefficient of friction

        ## Determine B (stiffness factor) using cornering and longitudinal stiffness (C_alpha and C_kappa)
        c_alpha_current_deg = self.cornering_stiffness_per_deg * ((f_z / self.f_z0) ** self.load_sensitivity_n)
        c_alpha_rad = c_alpha_current_deg * (180.0 / np.pi)

        b_lat =  c_alpha_rad / (self.params['y']['C'] * d) if d > 0 else 0
        self.params['y']['B'] = b_lat

        c_kappa_current = self.longitudinal_stiffness * ((f_z / self.f_z0) ** self.load_sensitivity_n)

        b_long = c_kappa_current / (self.params['x']['C'] * d) if d > 0 else 0
        self.params['x']['B'] = b_long

    def friction_circle(self, fx_raw, fy_raw, f_z):
        """
        Limits total force to the friction circle based on mu_peak and load

        :param fx_raw:
        :param fy_raw:
        :param f_z:
        :return fx_final, fy_final:
        """
        f_max = self.mu_peak * f_z
        vector_magnitude = np.sqrt(fx_raw ** 2 + fy_raw ** 2)

        # If total demand exceeds physical grip, scale back to the circle boundary
        if vector_magnitude > f_max and vector_magnitude > 0:
            scaling_factor = f_max / vector_magnitude
            fx_final = fx_raw * scaling_factor
            fy_final = fy_raw * scaling_factor
        else:
            fx_final = fx_raw
            fy_final = fy_raw

        return fx_final, fy_final