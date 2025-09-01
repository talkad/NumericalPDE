import numpy as np
import cv2


def get_exact_solution(hx=0.02, hy=0.02, ht=0.0001, T=2.0):
    """
    Compute exact solution on the same grid as numerical solution
    """
    x = np.arange(0, 1 + hx, hx)
    y = np.arange(0, 1 + hy, hy)
    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    nt = int(T / ht)
    
    U_frames = np.zeros((nt, nx, ny))
    
    for t in range(nt):
        current_time = t * ht
        U_frames[t] = (np.sin(np.pi * X)**2 + np.sin(np.pi * Y)**2) * np.exp(-np.pi * current_time)
        
        U_frames[t, 0, :] = U_frames[t, -1, :] = 0.0
        U_frames[t, :, 0] = U_frames[t, :, -1] = 0.0
    
    return U_frames


def visualize_solution(U_frames: np.ndarray, filename="output.mp4", fps: int = 10):
    """
    Create a video from solution frames
    """
    t, H, W = U_frames.shape[:3]
    
    # Normalize frames to 0-255 range
    U_min, U_max = np.min(U_frames), np.max(U_frames)
    if U_max > U_min:
        U_frames_norm = (U_frames - U_min) / (U_max - U_min) * 255
    else:
        U_frames_norm = np.zeros_like(U_frames)
    U_frames_norm = U_frames_norm.astype(np.uint8)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (W, H))
    
    for i in range(t):
        # Apply colormap
        frame = cv2.applyColorMap(U_frames_norm[i], cv2.COLORMAP_JET)
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {filename}")


def compute_source_term(x, y, t, eps):
    """
    Compute source term f(x,y,t) such that u_exact satisfies the PDE:
    ∂u/∂t = ε∆u + v⃗·∇u + f
    
    Therefore: f = ∂u/∂t - ε∆u - v⃗·∇u
    
    where u_exact(x,y,t) = (sin²(πx) + sin²(πy)) * exp(-πt)
    """
    
    # Precompute common terms
    sin_px = np.sin(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_px = np.cos(np.pi * x)
    cos_py = np.cos(np.pi * y)
    sin2_px = sin_px**2
    sin2_py = sin_py**2
    exp_neg_pit = np.exp(-np.pi * t)
    
    # Time derivative: ∂u/∂t
    du_dt = -np.pi * (sin2_px + sin2_py) * exp_neg_pit
    
    # Laplacian: ∆u = ∂²u/∂x² + ∂²u/∂y²
    # For u = sin²(πx) * exp(-πt):
    # ∂u/∂x = 2π sin(πx)cos(πx) * exp(-πt)
    # ∂²u/∂x² = 2π²[cos²(πx) - sin²(πx)] * exp(-πt) = 2π²cos(2πx) * exp(-πt)
    
    d2u_dx2 = 2 * np.pi**2 * np.cos(2 * np.pi * x) * exp_neg_pit
    d2u_dy2 = 2 * np.pi**2 * np.cos(2 * np.pi * y) * exp_neg_pit
    laplacian = d2u_dx2 + d2u_dy2
    
    # Gradient: ∇u = (∂u/∂x, ∂u/∂y)
    du_dx = 2 * np.pi * sin_px * cos_px * exp_neg_pit  # = π sin(2πx) * exp(-πt)
    du_dy = 2 * np.pi * sin_py * cos_py * exp_neg_pit  # = π sin(2πy) * exp(-πt)
    
    vx = x * (x - 2) * (1 - 2 * y)
    vy = -4 * y * (y - 1) * (1 - x)
    
    advection = vx * du_dx + vy * du_dy
    f = du_dt - eps * laplacian - advection
    
    return f


def calculate_error_norm(U_numerical, U_exact, hx, hy, ht):
    """
    Calculate the error norm as specified in the project
    """
    nt, nx, ny = U_numerical.shape
    
    error = 0.0
    for k in range(1, nt):  # k=1 to nt-1 (0-indexed)
        # Weight factor
        wk = 0.5 if k == nt - 1 else 1.0
        
        # Sum over interior points (exclude boundaries)
        time_error = 0.0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                time_error += abs(U_numerical[k, i, j] - U_exact[k, i, j])
        
        error += wk * ht * hx * hy * time_error
    
    return error