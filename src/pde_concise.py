"""
Steady-state solver for
  0 = (3/2) g + (x - I_{|y|<=z}) * d/dx g + (1/2) y d/dy g + (1/2) d^2/dy^2 g
  
Modified to use CENTERED differences in x (instead of upwind)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, bmat, eye, vstack, lil_matrix
from scipy.stats import norm
import osqp


def plot_dot_grid(x_centers, y_centers, z=None):
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')
    plt.figure(figsize=(10, 10))
    plt.scatter(Xc, Yc, s=4, c='black')
    if z is not None:
        plt.axhline(+z, color='red', linestyle='--')
        plt.axhline(-z, color='red', linestyle='--')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Grid Point Centers (dots)")
    plt.tight_layout()
    plt.show()

def z_from_delta(delta):
    """Compute z such that P(-z <= N(0,1) <= z) = 1 - delta."""
    return norm.ppf(1 - delta / 2)

# ==========================================================
# Parameters
# ==========================================================

Nx = 1000
Ny = 1000   

beta = 20.0
Y_MAX = 6

delta = 0.2
z = z_from_delta(delta)

eps_abs=1e-4
eps_rel=1e-4

print("Parameters:")
print(f"{Nx=}")
print(f"{Ny=}")
print(f"{beta=}")
print(f"{Y_MAX=}")

print(f"{delta=}")
print(f"{z=:.6f}")
print()

# ==========================================================
# x-grid
# ==========================================================

xi_uniform = np.linspace(0.0, 1.0, Nx + 1)
x_edges = 0.5 * (1 + np.tanh(beta * (xi_uniform - 0.5)) / np.tanh(beta / 2))
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

dx_back = np.empty(Nx)
dx_forw = np.empty(Nx)
dx_back[0] = x_centers[1] - x_centers[0]
dx_forw[-1] = x_centers[-1] - x_centers[-2]
dx_back[1:] = x_centers[1:] - x_centers[:-1]
dx_forw[:-1] = x_centers[1:] - x_centers[:-1]

w_x = x_edges[1:] - x_edges[:-1]

# ==========================================================
# y-grid 
# ==========================================================

y_edges = np.linspace(-Y_MAX, Y_MAX, Ny + 1)
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
w_y = np.diff(y_edges)

Yc = y_centers.copy()
wy_int = w_y.copy()
I_int = np.ones(Ny)

indicator = ((-z <= Yc) & (Yc <= z)).astype(float)

# examine grid
plot_dot_grid(x_centers, y_centers, z)

# ==========================================================
# Build y-operator Ly with one-sided second-order boundaries
# ==========================================================
# Ly discretizes: (1/2)(y * d/dy g + d^2/dy^2 g)

dy = y_centers[1] - y_centers[0]

print("Building Ly operator with one-sided 2nd-order boundary conditions...")

# Use sparse matrix construction with lil format for easier row-wise assembly
Ly = lil_matrix((Ny, Ny))

# Interior points: centered differences (j = 1, ..., Ny-2)
for j in range(1, Ny-1):
    yj = Yc[j]
    # First derivative (centered): (g_{j+1} - g_{j-1})/(2*dy)
    # Second derivative (centered): (g_{j+1} - 2*g_j + g_{j-1})/dy^2
    # Combined: (1/2)[y * dg/dy + d^2g/dy^2]
    c_minus = (0.5 - 0.25*dy*yj) / (dy*dy)
    c_plus  = (0.5 + 0.25*dy*yj) / (dy*dy)
    c_diag  = -1.0 / (dy*dy)
    
    Ly[j, j-1] = c_minus
    Ly[j, j]   = c_diag
    Ly[j, j+1] = c_plus

# Lower boundary (j=0): one-sided forward differences (2nd order)
# First derivative: (-3*g_0 + 4*g_1 - g_2)/(2*dy)
# Second derivative: (2*g_0 - 5*g_1 + 4*g_2 - g_3)/dy^2
j = 0
yj = Yc[j]
# Coefficients for (1/2)[y * dg/dy + d^2g/dy^2]
Ly[j, j]   = 0.5 * (yj * (-3.0)/(2*dy) + 2.0/(dy*dy))
Ly[j, j+1] = 0.5 * (yj * 4.0/(2*dy) - 5.0/(dy*dy))
Ly[j, j+2] = 0.5 * (yj * (-1.0)/(2*dy) + 4.0/(dy*dy))
Ly[j, j+3] = 0.5 * (-1.0/(dy*dy))

# Upper boundary (j=Ny-1): one-sided backward differences (2nd order)
# First derivative: (3*g_{N-1} - 4*g_{N-2} + g_{N-3})/(2*dy)
# Second derivative: (2*g_{N-1} - 5*g_{N-2} + 4*g_{N-3} - g_{N-4})/dy^2
j = Ny - 1
yj = Yc[j]
# Coefficients for (1/2)[y * dg/dy + d^2g/dy^2]
Ly[j, j]   = 0.5 * (yj * 3.0/(2*dy) + 2.0/(dy*dy))
Ly[j, j-1] = 0.5 * (yj * (-4.0)/(2*dy) - 5.0/(dy*dy))
Ly[j, j-2] = 0.5 * (yj * 1.0/(2*dy) + 4.0/(dy*dy))
Ly[j, j-3] = 0.5 * (-1.0/(dy*dy))

# Convert to CSR format for efficient operations
Ly = Ly.tocsr()

print(f"  Ly matrix: shape={Ly.shape}, nnz={Ly.nnz}")

# ==========================================================
# Build y-local block By 
# ==========================================================
Iy = diags(I_int, 0, shape=(Ny, Ny))
By = 1.5 * Iy + Ly 

# ==========================================================
# Build block operator A with centered differences in x
# ==========================================================
print("Building operator A with centered differences in x...")

blocks_main = []
blocks_low  = []
blocks_up   = []

for i in range(Nx):
    xi = x_centers[i]
    a = xi - indicator  # velocity coefficient at each y
    
    # Use centered differences: a * dg/dx ≈ a * (g_{i+1} - g_{i-1})/(dx_back + dx_forward)
    # Exception: boundaries use one-sided differences
    
    if i == 0:
        # Forward difference at left boundary: a * (g_1 - g_0)/dx_forward
        diag_contrib = diags(-a / dx_forw[i], 0)
        B_i = (By + diag_contrib).tocsr()
        blocks_main.append(B_i)
        
        L_i = diags(np.zeros(Ny), 0).tocsr()
        blocks_low.append(L_i)
        
        U_i = diags(a / dx_forw[i], 0).tocsr()
        blocks_up.append(U_i)
        
    elif i == Nx - 1:
        # Backward difference at right boundary: a * (g_{N-1} - g_{N-2})/dx_back
        diag_contrib = diags(a / dx_back[i], 0)
        B_i = (By + diag_contrib).tocsr()
        blocks_main.append(B_i)
        
        L_i = diags(-a / dx_back[i], 0).tocsr()
        blocks_low.append(L_i)
        
        U_i = diags(np.zeros(Ny), 0).tocsr()
        blocks_up.append(U_i)
        
    else:
        # Interior: centered difference a * (g_{i+1} - g_{i-1})/(dx_back + dx_forward)
        dx_cent = dx_back[i] + dx_forw[i]
        
        # No diagonal contribution for centered difference
        B_i = By.copy()
        blocks_main.append(B_i)
        
        L_i = diags(-a / dx_cent, 0).tocsr()
        blocks_low.append(L_i)
        
        U_i = diags(a / dx_cent, 0).tocsr()
        blocks_up.append(U_i)

A_blocks = []
for i in range(Nx):
    row_blocks = []
    for j in range(Nx):
        if j == i:
            row_blocks.append(blocks_main[i])
        elif j == i - 1:
            row_blocks.append(blocks_low[i])
        elif j == i + 1:
            row_blocks.append(blocks_up[i])
        else:
            row_blocks.append(None)
    A_blocks.append(row_blocks)

A = bmat(A_blocks, format='csr')

print("Assembled operator A:")
print(f"  size = {A.shape}, nnz = {A.nnz}, "
      f"sparsity = {100 * (1 - A.nnz / (A.shape[0] * A.shape[1])):.2f}%")

N_total = Nx * Ny

# ==========================================================
# Row weighting
# ==========================================================
W = np.outer(w_x, wy_int)
W_flat = W.ravel()
sqrtW = np.sqrt(W_flat)

A_w = A.multiply(sqrtW[:, None])
c = W_flat.copy()

# ==========================================================
# Build QP matrices
# ==========================================================
print("\nForming M = A_w^T A_w ...")
M = (A_w.T @ A_w).tocsr()
M = 0.5 * (M + M.T)
P = M.tocsc()
q = np.zeros(N_total)

Aeq = c.reshape(1, -1)
Aineq = eye(N_total, format='csc')
A_qp = vstack([Aeq, Aineq]).tocsc()

mass_target = 1.0
l = np.hstack([mass_target, np.zeros(N_total)])
u = np.hstack([mass_target, np.full(N_total, np.inf)])

diagM = M.diagonal()
print("diag(M): min =", diagM.min(), "max =", diagM.max())

# ==========================================================
# Solve QP 
# ==========================================================
print("\nSetting up and solving QP with OSQP...")
prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A_qp, l=l, u=u,
           verbose=True, 
           eps_abs=eps_abs, eps_rel=eps_rel, 
           max_iter=1_000_000)

res = prob.solve()
if res.info.status_val not in (1, 2):
    raise RuntimeError(f"OSQP did not find a valid solution. Status: {res.info.status}")

g = res.x
if np.any(g < 0):
    print(f"g contains negative entries: {np.min(g)=}")
    print(g)
g = np.maximum(g, 0.0)
mass = c @ g
if mass <= 0:
    raise RuntimeError("Nonpositive mass from QP solution; check formulation.")
g /= mass

print("\nQP solution stats:")
print(f"  min(g) = {g.min():.3e}, max(g) = {g.max():.3e}")
print(f"  integral(g) (FV quadrature) = {c @ g:.10f}")

# ==========================================================
# Postprocess & diagnostics
# ==========================================================
G = g.reshape(Nx, Ny)
Xc, Yc_grid = np.meshgrid(x_centers, Yc, indexing='ij')

residual = (A @ g).reshape(Nx, Ny)
L2_residual = np.sqrt(np.sum((residual**2) * W))
print(f"Weighted L2 residual (physical metric): {L2_residual:.6e}")
print(f"  min/max/mean g: {G.min():.3e} / {G.max():.3e} / {G.mean():.3e}")
print(f"  integral(g) (final check) = {np.sum(G * W):.10f}")

gy = np.sum(G * w_x[:, None], axis=0)
mass_y = np.sum(gy * w_y)
mean_y = np.sum(Yc * gy * w_y) / mass_y
var_y  = np.sum((Yc**2) * gy * w_y) / mass_y - mean_y**2
print(f"y-marginal: mass={mass_y:.6f}, mean={mean_y:.6e}, var={var_y:.6e}")
print(f"  EXPECTED: mass=1.0, mean=0.0, var=1.0")
print(f"  Variance error: {abs(var_y - 1.0):.6e}")

# ==========================================================
# Visualization
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
levels = np.linspace(0, G.max(), 20)
cf = ax.contourf(Xc, Yc_grid, G, levels=levels)
ax.axhline(z, ls='--', c='k', alpha=0.5)
ax.axhline(-z, ls='--', c='k', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Solution g(x,y) - CENTERED in x')
plt.colorbar(cf, ax=ax)

ax = axes[0, 1]
pm = ax.pcolormesh(Xc, Yc_grid, G, shading='auto')
ax.axhline(z, ls='--', c='k', alpha=0.5)
ax.axhline(-z, ls='--', c='k', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Solution (pcolormesh)')
plt.colorbar(pm, ax=ax)

ax = axes[1, 0]
gx = np.sum(G * wy_int[np.newaxis, :], axis=1)
ax.plot(x_centers, gx, lw=2)
ax.set_xlabel('x')
ax.set_ylabel('∫ g dy')
ax.set_title('Marginal in x')
ax.set_ylim(bottom=0.)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
y_fine = np.linspace(Yc[0], Yc[-1], 400)
normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * y_fine**2)
ax.plot(Yc, gy, lw=2, label='Computed')
ax.plot(y_fine, normal_pdf, 'r--', lw=2, alpha=0.7, label='N(0,1)')
ax.axvline(z, ls='--', c='k', alpha=0.4)
ax.axvline(-z, ls='--', c='k', alpha=0.4)
ax.set_xlabel('y')
ax.set_ylabel('∫ g dx')
ax.set_title('Marginal in y')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pde_solution_centered.png', dpi=150, bbox_inches='tight')
print("Saved: pde_solution_centered.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
levels = np.linspace(residual.min(), residual.max(), 21)
cf = ax.contourf(Xc, Yc_grid, residual, levels=levels, cmap='RdBu_r')
ax.axhline(z, ls='--', c='k', alpha=0.4)
ax.axhline(-z, ls='--', c='k', alpha=0.4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('PDE Residual')
plt.colorbar(cf, ax=ax)

ax = axes[1]
ax.hist(residual.ravel(), bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Residual value')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of residuals')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pde_residual_centered.png', dpi=150, bbox_inches='tight')
print("Saved: pde_residual_centered.png")

# Profile plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x_profile_values = [0.25, 0.5, 0.75]
colors = ['blue', 'green', 'red']

y_fine = np.linspace(Yc[0], Yc[-1], 400)
normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * y_fine**2)

for idx, x_val in enumerate(x_profile_values):
    ax = axes[idx]
    
    i_x = np.argmin(np.abs(x_centers - x_val))
    x_actual = x_centers[i_x]
    
    g_profile = G[i_x, :]
    integral_profile = np.sum(g_profile * wy_int)
    
    ax.plot(Yc, g_profile, lw=2.5, label=f'g(x={x_actual:.3f}, y)', 
            color=colors[idx], marker='o', markersize=3, markevery=10)
    ax.plot(y_fine, normal_pdf, 'k--', lw=2, alpha=0.6, label='N(0,1)')
    
    ax.axvline(z, ls=':', c='gray', alpha=0.5, label=f'y=±v ({z:.3f})')
    ax.axvline(-z, ls=':', c='gray', alpha=0.5)
    
    ax.set_xlabel('y', fontsize=11)
    ax.set_ylabel('g(x, y)', fontsize=11)
    ax.set_title(f'Profile at x = {x_actual:.3f}\n(integral = {integral_profile:.4f})', 
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([Yc[0], Yc[-1]])

plt.tight_layout()
plt.savefig('pde_profiles_centered.png', dpi=150, bbox_inches='tight')
print("Saved: pde_profiles_centered.png")
plt.show()

print("\n==== PROFILE VALUES AT y≈0 ====")
j_zero = np.argmin(np.abs(Yc))
print(f"y-index closest to 0: j={j_zero}, y={Yc[j_zero]:.6f}")
for x_val in x_profile_values:
    i_x = np.argmin(np.abs(x_centers - x_val))
    g_at_zero = G[i_x, j_zero]
    normal_at_zero = (1.0 / np.sqrt(2.0 * np.pi))
    ratio = g_at_zero / normal_at_zero
    print(f"x={x_centers[i_x]:.3f}: g(x,0)={g_at_zero:.6f}, "
          f"N(0,1) at 0={normal_at_zero:.6f}, ratio={ratio:.4f}")
print("================================\n")

k1 = 100
k2 = 50

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
gx = np.sum(G * wy_int[np.newaxis, :], axis=1)
axes[0].scatter(x_centers[:k1], gx[:k1], lw=2, s=1)
axes[0].set_xlabel('x')
axes[0].set_ylabel('∫ g dy')
axes[0].set_title('Marginal in x (1)')
axes[0].set_ylim(bottom=0.)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(x_centers[k1:-k2], gx[k1:-k2], lw=2, s=1)
axes[1].set_xlabel('x')
axes[1].set_title('Marginal in x (2)')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(x_centers[-k2:], gx[-k2:], lw=2, s=1)
axes[2].set_xlabel('x')
axes[2].set_title('Marginal in x (3)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()