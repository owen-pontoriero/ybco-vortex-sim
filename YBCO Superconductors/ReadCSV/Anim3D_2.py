import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# --- Grid ---
Nx, Ny = 150, 150
csv_filename = "data/psi_squared_data_3.csv"

x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)

# --- Animation settings ---
skip = 1
pause = 0.0001   # faster animation
norm = plt.Normalize(0, 1)

# --- Setup figure ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlim(0, 1)
ax.set_title("3D TDGL Vortex Animation")

# --- Initial surface ---
Z = np.zeros((Ny, Nx))
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',
    norm=norm,
    antialiased=False,   # <-- big speedup
    rstride=1, cstride=1 # <-- big speedup
)

plt.ion()
plt.show()

# --- FAST animation loop ---
with open(csv_filename) as f:
    reader = csv.reader(f)

    for frame_idx, row in enumerate(reader):

        if frame_idx % skip != 0:
            continue

        # convert row → array
        Z = np.array(row, dtype=float).reshape(Ny, Nx)
        Z = np.clip(np.nan_to_num(Z), 0, 1)

        # update surface
        surf.remove()
        surf = ax.plot_surface(
            X, Y, Z,
            cmap='viridis',
            norm=norm,
            antialiased=False,
            rstride=1, cstride=1
        )

        plt.pause(pause)

plt.ioff()
plt.show()