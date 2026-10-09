import matplotlib.pyplot as plt
import numpy as np
from Pacejka.tire import Tire

r55a = Tire("Hoosier R55A", mu_peak=1.6, )
r60b = Tire("Hoosier R60B", mu_peak=1.3)

# 2. Define operating state
vertical_load = 150.0  # lbs
slip_angles = np.linspace(-15, 15, 200)  # Smooth sweep from -15 to +15 degrees

# 3. Compute forces across the sweep
fy_r55a = [r55a.get_lateral_force(alpha, vertical_load) for alpha in slip_angles]
fy_r60b = [r60b.get_lateral_force(alpha, vertical_load) for alpha in slip_angles]

# 4. Plot the results using Matplotlib
plt.figure(figsize=(9, 5))

plt.plot(slip_angles, fy_r55a, label='Hoosier R55A (Soft, $\mu=1.6$)', color='crimson', linewidth=2.5)
plt.plot(slip_angles, fy_r60b, label='Hoosier R60B (Medium, $\mu=1.3$)', color='royalblue', linewidth=2.5)

plt.title(f'Pacejka Lateral Force Curves ($F_z = {vertical_load}$ lbs)', fontsize=12, fontweight='bold')
plt.xlabel('Slip Angle $\\alpha$ (degrees)', fontsize=10)
plt.ylabel('Lateral Force $F_y$ (lbs)', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.axvline(0, color='black', linewidth=0.8, linestyle='-')

plt.legend(fontsize=10)
plt.tight_layout()

# Display the interactive plot window
plt.show()
