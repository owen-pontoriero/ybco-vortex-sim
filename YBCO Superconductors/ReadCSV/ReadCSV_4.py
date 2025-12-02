import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Nx, Ny = 150, 150
csv_filename = "data/psi_squared_data_4.csv"
save_interval = 1000
dt = 0.0001

# Load CSV
data = np.loadtxt(csv_filename, delimiter=',')
n_frames = data.shape[0]

# Reshape all frames into 3D array (n_frames, Ny, Nx)
frames = data.reshape((n_frames, Ny, Nx))

# --- Create figure ---
fig, ax = plt.subplots(figsize=(6,6))
plt.pause(0.01)  # 50 ms between frames
im = ax.imshow(frames[0], extent=[0, Nx, 0, Ny], cmap="plasma", origin='lower')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("|ψ|²")
ax.set_xlabel("x (ξ)")
ax.set_ylabel("y (ξ)")

# --- Animation function ---
def update(frame_idx):
    im.set_data(frames[frame_idx])
    im.set_clim(np.min(frames[frame_idx]), np.max(frames[frame_idx]))  # auto scale
    ax.set_title(f"t = {frame_idx*save_interval*dt:.2f}")

ani = FuncAnimation(fig, update, frames=n_frames, interval=50)  # interval in ms
plt.show()
