class Tire:
    def __init__(self, name, mu_peak, cornering_stiffness_per_deg, load_sensitivity_n, c=1.4):
        self.name = name
        self.mu_peak = mu_peak
        self.cornering_stiffness_per_deg = cornering_stiffness_per_deg
        self.load_sensitivity_n = load_sensitivity_n
        self.c = c



    def coefficients_from_properties(self):

