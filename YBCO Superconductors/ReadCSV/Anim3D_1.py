import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# --- Grid parameters ---
Nx, Ny = 150, 150
Lx, Ly = 35.0, 35.0
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# --- CSV file ---
csv_filename = "data/psi_squared_data_3.csv"

# --- Animation settings ---
skip_frames = 1
pause_time = 0.005

# --- Plot setup ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("|ψ|²")

# FIX: lock color normalization
norm = plt.Normalize(vmin=0, vmax=1)

# Initial frame
Z = np.zeros((Ny, Nx))
surf = ax.plot_surface(X, Y, Z, cmap='viridis', norm=norm)

# --- Animation loop ---
with open(csv_filename, newline='') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        if idx % skip_frames != 0:
            continue

        Z = np.array(row, dtype=float).reshape(Ny, Nx)

        # Replace NaN or insane values
        Z = np.nan_to_num(Z, nan=0.0, posinf=1.0, neginf=0.0)
        Z = np.clip(Z, 0, 1)

        surf.remove()
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', norm=norm)

        plt.pause(pause_time)

plt.show()
