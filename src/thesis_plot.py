import time 

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.stats import norm


def build_transition_matrix(j_0, z, verbose=True):
    """
    Vectorized construction of P_epsilon.
    """
    t0 = time.perf_counter()
    epsilon = 1 / (2 * j_0) # unused
    eta = 1 / j_0
    gamma = 1 / np.sqrt(j_0)

    # Grid dimensions
    j_max = j_0
    k_max = int(np.ceil(6 * np.sqrt(j_0)))
    n_k = 2 * k_max + 1
    n_j = j_max + 1
    N = n_j * n_k

    j_idx = np.repeat(np.arange(n_j), n_k)
    k_idx = np.tile(np.arange(-k_max, k_max + 1), n_j)
    
    x = j_idx * eta
    y = k_idx * gamma

    current_idx = np.arange(N)
    
    in_region = np.abs(y) <= z
    
    rows = []
    cols = []
    data = []

    ### x transitions ###
    p_right = np.zeros(N)
    p_left  = np.zeros(N)
    p_stay_x = np.zeros(N)
    
    # case: |y| <= z  (forward difference)
    mask_fwd = in_region
    p_right[mask_fwd] = (1 - x[mask_fwd]) / 2
    p_stay_x[mask_fwd] = x[mask_fwd] / 2
    
    # case: |y| > z (backward difference)
    mask_bwd = ~in_region
    p_left[mask_bwd] = x[mask_bwd] / 2
    p_stay_x[mask_bwd] = 0.5 - x[mask_bwd] / 2

    # compute indices of neighbors (x direction)
    idx_right = current_idx + n_k
    valid_right = j_idx < j_max

    idx_left = current_idx - n_k
    valid_left = j_idx > 0
    
    # reflection at the boundary
    p_stay_x[mask_fwd & (~valid_right)] += p_right[mask_fwd & (~valid_right)]
    p_stay_x[mask_bwd & (~valid_left)] += p_left[mask_bwd & (~valid_left)]
    
    # right transitions
    mask_r_valid = mask_fwd & valid_right
    rows.append(current_idx[mask_r_valid])
    cols.append(idx_right[mask_r_valid])
    data.append(p_right[mask_r_valid])
    
    # left transitions
    mask_l_valid = mask_bwd & valid_left
    rows.append(current_idx[mask_l_valid])
    cols.append(idx_left[mask_l_valid])
    data.append(p_left[mask_l_valid])

    # ### y transitions ###
    p_up = 0.25 - (k_idx * eta) / 8
    p_dn = 0.25 + (k_idx * eta) / 8
    
    # compute indices of negihbors (y direction)
    idx_up = current_idx + 1
    valid_up = k_idx < k_max
    
    idx_dn = current_idx - 1
    valid_dn = k_idx > -k_max
    
    p_stay_x[~valid_up] += p_up[~valid_up]
    p_stay_x[~valid_dn] += p_dn[~valid_dn]
    
    # construct sparse matrix
    rows.append(current_idx[valid_up])
    cols.append(idx_up[valid_up])
    data.append(p_up[valid_up])
    
    rows.append(current_idx[valid_dn])
    cols.append(idx_dn[valid_dn])
    data.append(p_dn[valid_dn])
    
    rows.append(current_idx)
    cols.append(current_idx)
    data.append(p_stay_x)
    
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    
    P = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    state_list = [(j, k) for j in range(j_max + 1) for k in range(-k_max, k_max + 1)]
    index_of = {state: i for i, state in enumerate(state_list)}
    
    if verbose:
        print(f"Matrix built in {time.perf_counter() - t0:.4f}s")
    return P, state_list, index_of


def stationary_distribution(P, tol=1e-12, max_iter=150_000, verbose=True):
    # compute stationary distribution of P by Arnoldi iteration
    tstart = time.perf_counter()
    vals, vecs = eigs(P.T, k=1, which='LM', tol=tol, maxiter=max_iter)
    v = vecs[:, 0].real
    v = np.abs(v)
    v = v / np.sum(v)
    if verbose:
        print(f"Stationary distribution found in {time.perf_counter() - tstart:.2f}s")
    return v

def z_from_delta(delta):
    """Return z such that P(-z <= N(0,1) <= z) = 1 - delta."""
    return norm.ppf(1 - delta / 2)

def get_conditional_y_density(x_target, pi, j_0):
    eta = 1.0 / j_0
    gamma = 1.0 / np.sqrt(j_0)
    j_target = int(np.round(x_target / eta))
    j_target = max(0, min(j_0, j_target))
    effective_x = j_target * eta
    
    k_max = int(np.ceil(6 * np.sqrt(j_0)))
    
    n_k = 2 * k_max + 1
    start_idx = j_target * n_k
    end_idx = start_idx + n_k
    
    masses = pi[start_idx:end_idx]
    y_vals = np.arange(-k_max, k_max + 1) * gamma
            
    total_mass = np.sum(masses)
    return y_vals, (masses / total_mass) / gamma, effective_x


def get_x_density(j_0, state_list, pi):
    eta = 1 / j_0
    xs = np.array([j * eta for (j, k) in state_list])

    x_vals = np.unique(xs)
    x_marg = np.zeros_like(x_vals)
    for val, prob in zip(xs, pi):
        x_marg[np.where(x_vals == val)[0][0]] += prob
    
    return x_vals, x_marg

def get_stats(support, density):
    mean = np.sum(support * density)
    cumsum = np.cumsum(density)
    median = support[np.searchsorted(cumsum, 0.5)]
 
    return mean, median


if __name__ == "__main__":
    # Configure matplotlib to use Computer Modern font and 12pt size
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Computer Modern Roman']
    # plt.rcParams['font.size'] = 12
    # plt.rcParams['text.usetex'] = True


    j_0 = 5000
    delta = 0.1
    z = z_from_delta(delta)
    print(f"Grid parameters: j_0 = {j_0}, eta = {1/j_0:.6f}, gamma = {1/np.sqrt(j_0):.6f}")
    print(f"Confidence level 1-delta = {1-delta:.2f}, z = {z:.4f}")


    # Grid of plots for different confidence levels
    conf_levels = list(reversed([0.1, 0.2, 0.3, 0.45, 0.5, 0.55, 0.7, 0.8, 0.9]))
    conf_level = [.1]

    fig, axes = plt.subplots(3, 3, figsize=(8, 9))
    axes = axes.flatten()

    for ax, conf_level in zip(axes, conf_levels):
        z = z_from_delta(conf_level)
        P, state_list, index_of = build_transition_matrix(j_0, z)
        npz = np.load(f'ci_coverage/discretizations/coarse{j_0}_delta{delta}.npz')
        pi = npz['pi']

        mean, med = get_stats(*get_x_density(j_0, state_list, pi))

        # Extract x-coordinates and marginal
        xs = np.array([j * delta for (j, k) in state_list])
        x_vals = np.unique(xs)
        x_marg_prob = np.array([pi[xs == xv].sum() for xv in x_vals])
        x_marg_density = x_marg_prob / delta

        x_mean = np.sum(x_vals * x_marg_prob)
        x_cumsum = np.cumsum(x_marg_prob)
        x_median = x_vals[np.searchsorted(x_cumsum, 0.5)]

        ax.plot(x_vals, x_marg_density, label='_no_label')
        ax.axvline(x_mean, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x_median, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(rf"$1-\delta={1-conf_level:.2f}$")
        ax.set_xlabel("x")
        # ax.set_ylabel("density")
        ax.legend([f'mean={x_mean:.2f}', f'median={x_median:.2f}'], fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig('grid_discretization.png')
    plt.savefig('grid_discretization.pdf')
    plt.show()