import numpy as np
from utils import compute_source_term
from tqdm import tqdm

# Parameters
hx = hy = 0.02
ht = 0.00005
T = 0.5 # 2.0
eps = 1.0

x = np.arange(0, 1 + hx, hx)
y = np.arange(0, 1 + hy, hy)
nx, ny = len(x), len(y)
X, Y = np.meshgrid(x, y, indexing='ij')
nt = int(T / ht)

# Compute Velocity field
Vx = X * (X - 2) * (1 - 2 * Y)
Vy = -4 * Y * (Y - 1) * (1 - X)

ht_diff_max = (hx**2 * hy**2) / (2 * eps * (hx**2 + hy**2))

print(f"Grid size: {nx} x {ny}")
print(f"Time steps: {nt}")
print(f"CFL conditions to check:")
print(f"  Diffusion stability: ht <= {ht_diff_max:.6f}")

u = np.sin(np.pi * X)**2 + np.sin(np.pi * Y)**2
u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

u_new = np.zeros((nx, ny))
U_frames = np.zeros((nt, nx, ny))
U_frames[0] = u.copy()

for t in tqdm(range(1, nt)):
    u_new.fill(0.0)     

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            current_time = t * ht
            
            lap = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / hx**2 + \
                  (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / hy**2
            
            vx, vy = Vx[i, j], Vy[i, j]
            
            if vx >= 0:  
                if i >= 2:
                    ux = (3*u[i, j] - 4*u[i-1, j] + u[i-2, j]) / (2*hx)
                else:
                    ux = (u[i, j] - u[i-1, j]) / hx
            else:  
                if i <= nx - 3:
                    ux = (-3*u[i, j] + 4*u[i+1, j] - u[i+2, j]) / (2*hx)
                else:
                    ux = (u[i+1, j] - u[i, j]) / hx
 
            if vy >= 0:  
                if j >= 2:
                    uy = (3*u[i, j] - 4*u[i, j-1] + u[i, j-2]) / (2*hy)
                else:
                    uy = (u[i, j] - u[i, j-1]) / hy
            else: 
                if j <= ny - 3:
                    uy = (-3*u[i, j] + 4*u[i, j+1] + u[i, j+2]) / (2*hy)
                else:
                    uy = (u[i, j+1] - u[i, j]) / hy
            
            f = compute_source_term(x[i], y[j], current_time, eps)
            
            u_new[i, j] = u[i, j] + ht * (eps * lap - vx * ux - vy * uy + f)
    
    u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
    U_frames[t] = u_new.copy()
    
    u = u_new.copy()

np.save('U_scheme1_explicit.npy', U_frames[::50, :, :])