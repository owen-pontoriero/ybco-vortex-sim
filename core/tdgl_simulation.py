import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
import io
import imageio.v2 as imageio

class TDGLSimulation:
    def __init__(self,
                 kappa=5.0,
                 H=0.3,
                 dt=0.005,
                 resolution=100,
                 points_per_plot_unit=4,
                 psi_damping=1.0,
                 A_damping=8.0,
                 A_update_interval=3,
                 max_dA=0.1,
                 vortex_seeding=False,
                 evolve_A_flag=True,
                 initial_psi_amp=0.8,
                 noise_strength=0.05,
                 rng_seed=42):

        # --- GPU Acceleration Setup ---
        cp.random.seed(rng_seed)

        # --- Store Simulation Parameters ---
        self.kappa = kappa
        self.H = H
        self.dt = dt
        self.resolution = resolution
        self.points_per_plot_unit = points_per_plot_unit
        self.grid_size = resolution / points_per_plot_unit
        self.Lx = self.Ly = self.grid_size
        self.Nx = self.Ny = self.resolution
        
        # --- Store Damping & Control Parameters ---
        self.psi_damping = psi_damping
        self.A_damping = A_damping
        self.A_update_interval = A_update_interval
        self.max_dA = max_dA
        self.evolve_A_flag = evolve_A_flag
        self.vortex_seeding = vortex_seeding
        self.Initial_psi_amp = initial_psi_amp
        self.noise_strength = noise_strength

        # --- Setup Grid ---
        self.dx, self.dy, self.x, self.y, self.X, self.Y, self.KX, self.KY = self._setup_grid()

        # --- Initialize State Variables ---
        self.step_count = 0
        self.psi = cp.ones((self.Ny, self.Nx), dtype=complex) * self.Initial_psi_amp
        self.psi += self.noise_strength * (cp.random.randn(self.Ny, self.Nx) + 1j * cp.random.randn(self.Ny, self.Nx))
        
        # --- Magnetic vector potential in Landau gauge ---
        self.Ax = cp.zeros_like(self.X, dtype=cp.complex128)
        self.Ay = cp.array(self.H * self.X, dtype=cp.complex128)

        # --- (Optional) Seed Vortices ---
        if self.vortex_seeding:
            self._seed_vortices()
            
        print("TDGLSimulation instance created.")
        self.print_parameters()

    def print_parameters(self):
        print(f"Simulation Parameters:\nκ = {self.kappa}, dt = {self.dt}, H = {self.H}")
        print(f"Resolution: {self.resolution}, Points per plot unit: {self.points_per_plot_unit}")
        print(f"Vortex seeding is {'on' if self.vortex_seeding else 'off'}, and vector potential evolution is {'on' if self.evolve_A_flag else 'off'}")
        print(f"ψ damping: {self.psi_damping}")
        if self.evolve_A_flag:
            print(f"A damping: {self.A_damping}, A update interval: {self.A_update_interval}, Max change in A per timestep: {self.max_dA}")
        print(f"Initial Conditions: ψ_init = {self.Initial_psi_amp}, noise = {self.noise_strength}\n")

    def _setup_grid(self):
        """Creates the spatial and Fourier-space grids."""

        x = cp.linspace(0, self.Lx, self.Nx, endpoint=True)
        y = cp.linspace(0, self.Ly, self.Ny, endpoint=True)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X, Y = cp.meshgrid(x, y)

        kx = 2 * cp.pi * cp.fft.fftfreq(self.Nx, dx)
        ky = 2 * cp.pi * cp.fft.fftfreq(self.Ny, dy)

        KX, KY = cp.meshgrid(kx, ky)

        return dx, dy, x, y, X, Y, KX, KY

    def _seed_vortices(self):
        """Initializes psi with a hexagonal vortex lattice."""
        print("Seeding vortices...")
        vortex_positions = []
        core_size_relative = 1.5  # Arbitrary vortex scale
        
        # Estimate vortex spacing for hexagonal lattice
        a_vortex = cp.sqrt(2 * cp.sqrt(3) / self.H)
        n_x = int((self.Lx / a_vortex).item()) + 1
        n_y = int((self.Ly / (a_vortex * cp.sqrt(3) / 2)).item()) + 1

        # Create hexagonal lattice
        for j in range(n_y):
            for i in range(n_x):
                x_pos = i * a_vortex + 0.5 * a_vortex * (j % 2)
                y_pos = j * a_vortex * cp.sqrt(3) / 2
                
                if 0.5 < x_pos < self.Lx - 0.5 and 0.5 < y_pos < self.Ly - 0.5:
                    vortex_positions.append((x_pos, y_pos))

        # Seed vortices
        for vx, vy in vortex_positions:
            phase_wind = cp.arctan2(self.Y - vy, self.X - vx)
            dist = cp.sqrt((self.X - vx)**2 + (self.Y - vy)**2)
            
            vortex_profile = cp.tanh(dist / (core_size_relative * self.dx))
            self.psi *= vortex_profile * cp.ecp(1j * phase_wind)
        
        print(f"Seeded: {len(vortex_positions)} vortices in hexagonal pattern\n")

    # --- Core Computational Methods (Internal) ---

    def _laplacian_cov_2D_fft(self, psi, Ax, Ay):
        """Covariant 2d Laplacian with FFTs"""

        psi_k = cp.fft.fft2(psi)

        dpsi_dx = cp.fft.ifft2(1j * self.KX * psi_k)
        dpsi_dy = cp.fft.ifft2(1j * self.KY * psi_k)

        Dx_psi = dpsi_dx - 1j * Ax * psi
        Dy_psi = dpsi_dy - 1j * Ay * psi
        
        Dx_psi_k = cp.fft.fft2(Dx_psi)
        Dy_psi_k = cp.fft.fft2(Dy_psi)

        dDx_dx = cp.fft.ifft2(1j * self.KX * Dx_psi_k)
        dDy_dy = cp.fft.ifft2(1j * self.KY * Dy_psi_k)

        Dxx_psi = dDx_dx - 1j * Ax * Dx_psi
        Dyy_psi = dDy_dy - 1j * Ay * Dy_psi
        
        return Dxx_psi + Dyy_psi

    def _covariant_derivative_2D_fft(self, psi, Ax, Ay):
        """Covariant derivative in 2D using FFTs"""

        psi_k = cp.fft.fft2(psi)

        dpsi_dx = cp.fft.ifft2(1j * self.KX * psi_k)
        dpsi_dy = cp.fft.ifft2(1j * self.KY * psi_k)
        
        return dpsi_dx - 1j * Ax * psi, dpsi_dy - 1j * Ay * psi

    def _supercurrent(self, psi, Ax, Ay):
        """Calculate supercurrent density J = Im[ψ*(D_μ ψ)]"""
        Dx_psi, Dy_psi = self._covariant_derivative_2D_fft(psi, Ax, Ay)
        Jx = cp.imag(cp.conj(psi) * Dx_psi)
        Jy = cp.imag(cp.conj(psi) * Dy_psi)
        
        return Jx, Jy

    def _curl_2D(self, Ax, Ay):
        """2D Curl using FFTs"""
        
        Ax_k = cp.fft.fft2(Ax)
        Ay_k = cp.fft.fft2(Ay)

        dAx_dy = cp.fft.ifft2(1j * self.KY * Ax_k)
        dAy_dx = cp.fft.ifft2(1j * self.KX * Ay_k)

        return dAy_dx - dAx_dy

    def _evolve_psi(self):
        """Evolve order parameter, updating self.psi in-place."""
        
        nonlinear_term = (1 - cp.abs(self.psi)**2) * self.psi
        laplacian_term = self._laplacian_cov_2D_fft(self.psi, self.Ax, self.Ay)

        dpsi_dt = (nonlinear_term + laplacian_term) / self.psi_damping
        self.psi += self.dt * dpsi_dt
    
    def _evolve_A(self):
        """Evolve vector potential, updating self.Ax/Ay in-place."""

        Jx, Jy = self._supercurrent(self.psi, self.Ax, self.Ay)
        
        Bz = self._curl_2D(self.Ax, self.Ay)
        
        Bz_k = cp.fft.fft2(Bz)
        dBz_dx = cp.fft.ifft2(1j * self.KX * Bz_k)
        dBz_dy = cp.fft.ifft2(1j * self.KY * Bz_k)
        
        dAx_dt = (Jx - self.kappa**2 * dBz_dy) / self.A_damping
        dAy_dt = (Jy + self.kappa**2 * dBz_dx - self.H) / self.A_damping

        eps = 1e-10
        dAx_dt_norm = cp.abs(dAx_dt) + eps
        dAy_dt_norm = cp.abs(dAy_dt) + eps

        scale_x = cp.minimum(1, self.max_dA / dAx_dt_norm)
        scale_y = cp.minimum(1, self.max_dA / dAy_dt_norm)
        
        dt_A = self.dt * self.A_update_interval
        
        self.Ax += dt_A * dAx_dt * scale_x
        self.Ay += dt_A * dAy_dt * scale_y

    # --- Public API Methods ---

    def update_one_step(self):
        """Update by one step. ψ is always updated, and A is updated periodically, based on simulation setup"""
        self._evolve_psi()
        
        if self.evolve_A_flag and self.step_count % self.A_update_interval == 0:
            self._evolve_A()
        
        self.step_count += 1
        return self.step_count

    def set_parameters(self, H=None, kappa=None, psi_damping=None, A_damping=None):
        """Safely updated simulation parameters in real time"""
        if H is not None:
            self.H = H
        if kappa is not None:
            self.kappa = kappa
        if psi_damping is not None:
            self.psi_damping = psi_damping
        if A_damping is not None:
            self.A_damping = A_damping

    def get_density(self):
        """Returns the current superconducting density |ψ|² for plotting"""
        return cp.abs(self.psi)**2

    def get_phase(self):
        """Returns the current phase of ψ for plotting"""
        return cp.angle(self.psi)
    
    def get_current_time(self):
        """Returns the current simulation time"""
        return self.step_count * self.dt
    
    def check_instability(self):
        """Checks for non-physical values"""
        max_psi = cp.max(cp.abs(self.psi))
        if cp.isnan(max_psi) or max_psi > 2.0:
            return True
        return False
    
    def to_gpu(self, arr):
        """Move array to GPU"""
        if isinstance(arr, cp.ndarray):
            return cp.asarray(arr)
        return arr
    
    def to_cpu(self, arr):
        """Move array to CPU"""
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def render_frame(self):
        """Render a simulation frame"""
        density = cp.asnumpy(cp.abs(self.psi)**2)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        im = ax.imshow(density, cmap='viridis', origin='lower')
        ax.set_axis_off()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        buf.seek(0)

        return imageio.imread(buf)
    
n_steps = 40000
save_interval = 4000
frame_interval = 30

sim = TDGLSimulation(kappa=5.0,
                H=0.3,
                dt=0.002,
                resolution=100,
                points_per_plot_unit=4,
                psi_damping=1.0,
                A_damping=8.0,
                A_update_interval=3,
                max_dA=0.1,
                vortex_seeding=False,
                evolve_A_flag=True,
                initial_psi_amp=0.8,
                noise_strength=0.05,
                rng_seed=45
)

psi_history = []
A_history = []
time_points = []
frames = []

print("Starting Simulation...")
start = time.time()
last_save_time = start
last_save_step = 0

for step in range(n_steps):
    
    sim.update_one_step()
    
    if step % save_interval == 0:
        psi_history.append(sim.to_cpu(sim.get_density()))
        A_history.append((sim.to_cpu(sim.Ax.copy()), sim.to_cpu(sim.Ay.copy())))
        t = sim.get_current_time()
        time_points.append(t)
        
        density = psi_history[-1]
        max_psi = np.max(np.sqrt(density))
        min_psi = np.min(np.sqrt(density))
        avg_psi = np.mean(np.sqrt(density))

        try:
            avg_dt_rt = (time.time() - last_save_time) / (step - last_save_step) * 1e3
        except Exception as e:
            avg_dt_rt = 0
        
        print(f"t={t:.1f} | |ψ|: min={min_psi:.3f}, avg={avg_psi:.3f}, max={max_psi:.3f} | Time elapsed for steps {last_save_step} to {step}: {time.time() - last_save_time:.2f}s | Average time to process one step: {avg_dt_rt}ms")
        last_save_step = step
        last_save_time = time.time()
        
        if sim.check_instability():
            print("Instability detected! Stopping simulation.")
            break

    if step % frame_interval == 0:
        frames.append(sim.render_frame())