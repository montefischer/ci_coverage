import time 

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigs
from scipy.stats import norm

def logit(x):
    return np.log(np.divide(x, np.ones_like(x) - x))

def logit_inv(y):
    return np.reciprocal(1 + np.exp(-y))

def logit_prime_inv(x):
    return 2 * (np.cosh(x) + 1)

def tangent(x):
    return np.tan(np.pi*(x - 0.5))

def tangent_inv(y):
    return 0.5 + np.arctan(y) / np.pi

def tangent_prime_inv(x):
    return (np.pi * (1 + x**2))


def compute_grid_parameters(j0, xi_bar, f_inv, f_prime_inv):
    """
    Compute consistent grid parameters for the discretization.

    Parameters:
    -----------
    j0 : int
        Number of x discretization points
    xi_bar : float
        x-coordinate range [-xi_bar, xi_bar]
    f_inv : callable
        Inverse transformation f^(-1)
    f_prime_inv : callable
        Derivative f'(f^(-1)(ξ))

    Returns:
    --------
    eta : float
        Grid spacing in x direction
    epsilon : float
        Time step parameter
    gamma : float
        Grid spacing in y direction
    k_max : int
        Maximum k index for y discretization
    """
    eta = xi_bar / j0
    j_max = j0

    # Compute epsilon from the derivative
    x_grid = eta * np.arange(-j_max, j_max + 1)
    epsilon = eta

    # Compute gamma and k_max
    gamma = np.sqrt(2 * epsilon)
    k_max = int(np.ceil(10 / gamma))

    return eta, epsilon, gamma, k_max


def to_rate_matrix(P):
    """
    Convert sparse off-diagonal rate matrix to proper rate matrix Q where Q_ii = -sum_{j≠i} Q_ij.

    Assumes P contains only off-diagonal rates. Adds diagonal entries to ensure row sums = 0.
    """
    from scipy.sparse import csr_array
    import numpy as np

    P_csr = P.tocsr()
    n = P_csr.shape[0]

    # Compute off-diagonal row sums (current matrix has no diagonal)
    off_diag_sums = np.array(P_csr.sum(axis=1)).flatten()

    # Build new COO format with diagonal entries added
    rows_list = []
    cols_list = []
    data_list = []

    # Copy all off-diagonal entries
    for i in range(n):
        row_start = P_csr.indptr[i]
        row_end = P_csr.indptr[i + 1]

        for idx in range(row_start, row_end):
            j = P_csr.indices[idx]
            rows_list.append(i)
            cols_list.append(j)
            data_list.append(P_csr.data[idx])

    # Add diagonal entries: Q_ii = -sum_{j≠i} Q_ij
    for i in range(n):
        rows_list.append(i)
        cols_list.append(i)
        data_list.append(-off_diag_sums[i])

    rows = np.array(rows_list)
    cols = np.array(cols_list)
    data = np.array(data_list)

    Q = csr_array((data, (rows, cols)), shape=(n, n))

    return Q



def build_transition_matrix(j0, xi_bar, z, f, f_inv, f_prime_inv, verbose=True):
    """
    Vectorized construction of P_epsilon.

    j0: number of x discretization points
    xi_bar: [-xi_bar, xi_bar] is x-dimensional discretization
    z: parameter of Z(1)
    f: nonlinear map
    f_inv: inverse of nonlinear map
    f_prime_inv: derivative of f composed with f_inv
    """
    t0 = time.perf_counter()

    # Compute grid parameters consistently
    eta, epsilon, gamma, k_max = compute_grid_parameters(j0, xi_bar, f_inv, f_prime_inv)

    # Grid dimensions
    j_max = j0
    n_j = 2 * j_max + 1
    n_k = 2 * k_max + 1
    N = n_j * n_k

    print(f"{eta=}")
    print(f"{epsilon=}")
    print(f"{gamma=}")
    print(f"{k_max=}")
    print(f"{n_j=}")
    print(f"{n_k=}")
    print(f"{N=}")

    j_idx = np.repeat(np.arange(-j_max, j_max + 1), n_k)
    k_idx = np.tile(np.arange(-k_max, k_max + 1), n_j)

    x = j_idx * eta
    y = k_idx * gamma

    current_idx = np.arange(N)
    
    in_region = np.abs(y) <= z
    
    rows = []
    cols = []
    data = []

    ### x transitions ###
    q_right = np.zeros(N)
    q_left  = np.zeros(N)
    q_stay_x = np.zeros(N)
    
    # case: |y| <= z  (forward difference)
    mask_fwd = in_region

    print("num in region",np.sum(in_region), "/", len(in_region))

    q_right[mask_fwd] = f_prime_inv(x[mask_fwd]) * (1 - f_inv(x[mask_fwd]))
    
    # case: |y| > z (backward difference)
    mask_bwd = ~in_region
    q_left[mask_bwd] = f_prime_inv(x[mask_bwd]) * f_inv(x[mask_bwd]) 
    # print('max bwd derivative',)

    # compute indices of neighbors (x direction)
    idx_right = current_idx + n_k
    valid_right = j_idx < j_max

    idx_left = current_idx - n_k
    valid_left = j_idx > -j_max
    
    # right transitions
    mask_r_valid = mask_fwd & valid_right
    rows.append(current_idx[mask_r_valid])
    cols.append(idx_right[mask_r_valid])
    data.append(q_right[mask_r_valid])
    
    # left transitions
    mask_l_valid = mask_bwd & valid_left
    rows.append(current_idx[mask_l_valid])
    cols.append(idx_left[mask_l_valid])
    data.append(q_left[mask_l_valid])

    # ### y transitions ###
    q_up = 2 - (k_idx * epsilon) / 4
    q_dn = 2 + (k_idx * epsilon) / 4
    
    # compute indices of negihbors (y direction)
    idx_up = current_idx + 1
    valid_up = k_idx < k_max
    
    idx_dn = current_idx - 1
    valid_dn = k_idx > -k_max
    
    # construct sparse matrix
    rows.append(current_idx[valid_up])
    cols.append(idx_up[valid_up])
    data.append(q_up[valid_up])
    
    rows.append(current_idx[valid_dn])
    cols.append(idx_dn[valid_dn])
    data.append(q_dn[valid_dn])
    
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    
    Q = csr_array((data, (rows, cols)), shape=(N, N))
    Q = to_rate_matrix(Q)


    state_list = [(j, k) for j in range(-j_max, j_max + 1) for k in range(-k_max, k_max + 1)]
    index_of = {state: i for i, state in enumerate(state_list)}

    if verbose:
        print(f"Matrix built in {time.perf_counter() - t0:.4f}s")
    return Q, state_list, index_of



def stationary_distribution(Q, tol=1e-12, max_iter=150_000, verbose=True):
    """
    Compute stationary distribution of CTMC by solving π^T Q = 0 with ∑π_i = 1.

    Uses GMRES iterative solver with high precision:
    Replace one equation in Q^T π = 0 with the normalization constraint.
    """
    from scipy.sparse.linalg import gmres

    tstart = time.perf_counter()

    Q_csr = Q.tocsr()
    n = Q_csr.shape[0]

    # Set up linear system: Q^T π = 0 with normalization constraint
    # Replace the first row of Q^T with ones (for normalization constraint)
    Q_T = Q_csr.T.tolil()  # Convert to lil for efficient row modification
    Q_T[0, :] = 1.0

    # Right-hand side: [1, 0, 0, ..., 0]^T
    b = np.zeros(n)
    b[0] = 1.0

    # Solve Q_T_modified @ π = b using GMRES
    Q_T_modified = Q_T.tocsr()

    # Use uniform distribution as initial guess
    x0 = np.ones(n) / n

    global it
    it = 0
    def callback(x):
        global it
        it += 1
        if it % 1000 == 0:
            print(it)

    pi, info = gmres(Q_T_modified, b, x0=x0, rtol=tol, atol=tol, maxiter=max_iter, restart=50, callback=callback)

    if info != 0:
        print(f"Warning: GMRES did not converge. Info code: {info}")

    # Ensure non-negativity (numerical errors might produce small negatives)
    pi = np.abs(pi)
    pi = pi / np.sum(pi)  # Renormalize

    if verbose:
        print(f"Stationary distribution found in {time.perf_counter() - tstart:.2f}s")
        # Verify π^T Q ≈ 0
        residual = np.abs(Q_csr.T @ pi).max()
        print(f"Max residual |π^T Q|: {residual:.2e}")

    return pi

def stationary_distribution_arnoldi(Q, k=1, verbose=True):
    """
    Compute stationary distribution of CTMC using Arnoldi iteration.

    Finds the eigenvector corresponding to eigenvalue λ=0 of the rate matrix Q.
    For a rate matrix Q, the stationary distribution π satisfies:
    - Q^T π = 0 (equivalently, π is the right eigenvector of Q^T for λ=0)
    - ∑π_i = 1
    - π_i ≥ 0

    Parameters:
    -----------
    Q : sparse matrix
        Rate matrix (generator) of CTMC. Should have row sums equal to 0.
    k : int
        Number of eigenvalues to compute. Default is 1 (only the zero eigenvalue).
        Increase if convergence issues occur.
    verbose : bool
        If True, print timing and diagnostic information.

    Returns:
    --------
    pi : ndarray
        Stationary distribution (probability vector summing to 1)
    """
    tstart = time.perf_counter()

    Q_csr = Q.tocsr()
    n = Q_csr.shape[0]

    # Find eigenvalues closest to 0 of Q^T (which gives us left eigenvectors of Q)
    # For a rate matrix, λ=0 is always an eigenvalue
    # We use sigma=0 to find eigenvalues near 0
    # v0 can be used to provide initial vector (using uniform distribution as hint)
    v0 = np.ones(n) / np.sqrt(n)

    try:
        # Find k eigenvalues closest to 0 and their corresponding eigenvectors
        # Q^T v = λ v, so we want the eigenvector for λ ≈ 0
        eigenvalues, eigenvectors = eigs(Q_csr.T, k=k, sigma=0, which='LM', v0=v0)

        # Find the eigenvalue closest to 0
        idx = np.argmin(np.abs(eigenvalues))
        lambda_0 = eigenvalues[idx]
        pi = eigenvectors[:, idx].real  # Take real part (should be real for rate matrix)

        if verbose:
            print(f"Found eigenvalue λ = {lambda_0:.2e} (should be ≈ 0)")
            print(f"Eigenvalue is {'real' if np.abs(eigenvalues[idx].imag) < 1e-10 else 'complex'}")

    except Exception as e:
        if verbose:
            print(f"Warning: eigs failed with error: {e}")
            print("Falling back to computing more eigenvalues...")

        # Fallback: try with more eigenvalues or different parameters
        try:
            eigenvalues, eigenvectors = eigs(Q_csr.T, k=min(k+4, n-2), sigma=0, which='LM')
            idx = np.argmin(np.abs(eigenvalues))
            lambda_0 = eigenvalues[idx]
            pi = eigenvectors[:, idx].real

            if verbose:
                print(f"Found eigenvalue λ = {lambda_0:.2e} with fallback method")
        except Exception as e2:
            raise RuntimeError(f"Both eigs attempts failed: {e}, {e2}")

    # Ensure non-negativity (eigenvector sign is arbitrary, and numerical errors might give small negatives)
    pi = np.abs(pi)

    # Normalize to probability distribution (sum to 1)
    pi = pi / np.sum(pi)

    if verbose:
        print(f"Stationary distribution found in {time.perf_counter() - tstart:.2f}s")
        # Verify π^T Q ≈ 0
        residual = np.abs(Q_csr.T @ pi).max()
        print(f"Max residual |π^T Q|: {residual:.2e}")

    return pi

def z_from_delta(delta):
    """Return z such that P(-z <= N(0,1) <= z) = 1 - delta."""
    return norm.ppf(1 - delta / 2)

def get_conditional_y_density(x_target, pi, j0, xi_bar, f_inv, f_prime_inv):
    # Compute grid parameters consistently
    eta, epsilon, gamma, k_max = compute_grid_parameters(j0, xi_bar, f_inv, f_prime_inv)

    j_max = j0
    j_target = int(np.round(x_target / eta))
    j_target = max(-j_max, min(j_max, j_target))
    effective_x = j_target * eta

    n_k = 2 * k_max + 1
    # Account for centered grid: j ranges from -j_max to j_max
    start_idx = (j_target + j_max) * n_k
    end_idx = start_idx + n_k

    masses = pi[start_idx:end_idx]
    y_vals = np.arange(-k_max, k_max + 1) * gamma

    total_mass = np.sum(masses)
    return y_vals, (masses / total_mass) / gamma, effective_x

def get_x_density(j0, xi_bar, state_list, pi):
    eta = xi_bar / j0
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

def transform_density_to_original(xi_vals, xi_density, f_inv, f_prime_inv):
    """
    Transform density from ξ space back to X space.

    Given a density p_ξ(ξ) where ξ = f(X), compute the density p_X(x).

    Uses the change of variables formula:
    p_X(x) = p_ξ(f(x)) * |f'(x)| = p_ξ(ξ) * |f'(f^(-1)(ξ))|

    Parameters:
    -----------
    xi_vals : array
        Grid points in ξ space
    xi_density : array
        Density values in ξ space (same length as xi_vals)
    f_inv : callable
        Inverse transformation f^(-1): ξ -> X
    f_prime_inv : callable
        Derivative f'(f^(-1)(ξ))

    Returns:
    --------
    x_vals : array
        Grid points in original X space
    x_density : array
        Density values in X space
    """
    # Transform grid points to original space
    x_vals = f_inv(xi_vals)

    # Apply change of variables formula
    # p_X(x) = p_ξ(ξ) * |f'(x)| = p_ξ(ξ) / |df^(-1)/dξ|
    # Since f'(f^(-1)(ξ)) gives us f'(x), and we need 1/|(f^(-1))'(ξ)|
    # Note: (f^(-1))'(ξ) = 1/f'(f^(-1)(ξ))
    # So: p_X(x) = p_ξ(ξ) * f'(f^(-1)(ξ))

    jacobian = f_prime_inv(xi_vals)
    x_density = xi_density * jacobian

    # Sort by x values (since transformation may reorder)
    sort_idx = np.argsort(x_vals)
    x_vals = x_vals[sort_idx]
    x_density = x_density[sort_idx]

    return x_vals, x_density

def bin_density_to_original_space(xi_vals, xi_probs, f_inv, n_bins=500, x_range=None):
    """
    Bin probability mass from ξ space into evenly spaced bins in X space.

    This is more appropriate than transform_density_to_original when you want
    uniform resolution in the original space rather than following the transformed grid.

    Parameters:
    -----------
    xi_vals : array
        Grid points in ξ space
    xi_probs : array
        Probability masses at each ξ grid point (not density)
    f_inv : callable
        Inverse transformation f^(-1): ξ -> X
    n_bins : int
        Number of bins in X space
    x_range : tuple or None
        (x_min, x_max) for binning. If None, determined from transformed values.

    Returns:
    --------
    x_bin_centers : array
        Center of each bin in X space
    x_density : array
        Density in each bin (probability mass / bin width)
    """
    # Transform ξ points to X space
    x_transformed = f_inv(xi_vals)

    # Determine X range for binning
    if x_range is None:
        # Add small margin to ensure all points are included
        x_min = np.min(x_transformed)
        x_max = np.max(x_transformed)
        margin = 0.01 * (x_max - x_min)
        x_min -= margin
        x_max += margin
    else:
        x_min, x_max = x_range

    # Create even bins in X space
    x_bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    x_bin_width = x_bin_edges[1] - x_bin_edges[0]
    x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2

    # Initialize bin counts
    x_bin_probs = np.zeros(n_bins)

    # Bin each transformed point
    for x_val, prob in zip(x_transformed, xi_probs):
        # Find which bin this point falls into
        bin_idx = np.searchsorted(x_bin_edges[1:], x_val)
        if 0 <= bin_idx < n_bins:
            x_bin_probs[bin_idx] += prob

    # Convert probability mass to density
    x_density = x_bin_probs / x_bin_width

    return x_bin_centers, x_density

def simulate_paths(Q, state_list, n_paths=5, T_max=10.0, seed=None):
    """
    Simulate sample paths from the continuous-time Markov chain.

    Parameters:
    -----------
    Q : sparse matrix
        Rate matrix (generator) of CTMC
    state_list : list of tuples
        List of (j, k) states
    n_paths : int
        Number of paths to simulate
    T_max : float
        Maximum simulation time for each path
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    paths : list of arrays
        Each array is shape (n_transitions, 2) containing (j, k) coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(state_list)
    paths = []

    # Keep Q sparse and convert to CSR format for efficient row access
    Q_csr = Q.tocsr()

    for _ in range(n_paths):
        # Start from a random state (or could start from center)
        current_state = np.random.randint(N)
        path = []
        t_current = 0.0

        while t_current < T_max:
            j, k = state_list[current_state]
            path.append([j, k])

            # Extract only non-zero entries from the current row (sparse sampling)
            row_start = Q_csr.indptr[current_state]
            row_end = Q_csr.indptr[current_state + 1]

            # Get non-zero column indices and their rates
            nonzero_cols = Q_csr.indices[row_start:row_end]
            nonzero_rates = Q_csr.data[row_start:row_end]

            # Find exit rate (negative of diagonal)
            exit_rate = 0.0
            off_diag_cols = []
            off_diag_rates = []
            for idx, col in enumerate(nonzero_cols):
                if col == current_state:
                    exit_rate = -nonzero_rates[idx]
                else:
                    off_diag_cols.append(col)
                    off_diag_rates.append(nonzero_rates[idx])

            if exit_rate <= 0:
                # Absorbing state, stop
                break

            # Sample holding time (exponential with rate = exit_rate)
            holding_time = np.random.exponential(1.0 / exit_rate)
            t_current += holding_time

            if t_current >= T_max:
                break

            # Sample next state from off-diagonal rates
            if len(off_diag_cols) > 0:
                off_diag_rates = np.array(off_diag_rates)
                transition_probs = off_diag_rates / exit_rate
                current_state = np.random.choice(off_diag_cols, p=transition_probs)
            else:
                # No transitions possible, stop
                break

        paths.append(np.array(path))

    return paths

def plot_sample_paths(paths, j0, xi_bar, z, f_inv, f_prime_inv, title="Sample Paths on 2D Grid"):
    """
    Plot sample paths on the (x, y) grid.

    Parameters:
    -----------
    paths : list of arrays
        Output from simulate_paths
    j0 : int
        Grid parameter
    xi_bar : float
        x-coordinate range
    z : float
        Confidence region boundary
    f_inv : callable
        Inverse transformation f^(-1)
    f_prime_inv : callable
        Derivative f'(f^(-1)(ξ))
    title : str
        Plot title
    """
    # Compute grid parameters consistently
    eta, epsilon, gamma, k_max = compute_grid_parameters(j0, xi_bar, f_inv, f_prime_inv)

    plt.figure(figsize=(12, 8))

    # Plot the confidence region boundaries
    j_max = j0
    x_range = np.array([-j_max, j_max]) * eta
    plt.xlim(-xi_bar - 0.1, xi_bar + 0.1)
    plt.axhline(y=z, color='red', linestyle='--', alpha=0.5, linewidth=2, label=f'|y| = {z:.2f}')
    plt.axhline(y=-z, color='red', linestyle='--', alpha=0.5, linewidth=2)

    # Plot each path
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
    for i, path in enumerate(paths):
        x_coords = path[:, 0] * eta
        y_coords = path[:, 1] * gamma
        plt.plot(x_coords, y_coords, alpha=0.7, linewidth=1, color=colors[i], label=f'Path {i+1}')
        plt.scatter(x_coords[0], y_coords[0], s=100, marker='o', color=colors[i], edgecolors='black', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], s=100, marker='s', color=colors[i], edgecolors='black', zorder=5)

    plt.xlabel(r'$\xi$ (transformed x-coordinate)', fontsize=12)
    plt.ylabel(r'$y$ (Brownian motion)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    return plt.gcf()

if __name__ == "__main__":
    j0 = 3_000
    delta = 0.5
    z = z_from_delta(delta)

    print(f"Grid parameters: j_0 = {j0}, eta = {1/j0:.6f}, gamma = {1/np.sqrt(j0):.6f}")
    print(f"Confidence level 1-delta = {1-delta:.2f}, z = {z:.4f}")

    xi_bar = 20

    # f, f_inv, f_prime_inv = tangent, tangent_inv, tangent_prime_inv
    f, f_inv, f_prime_inv = logit, logit_inv, logit_prime_inv

    # Q, state_list, index_of = build_transition_matrix(j0, xi_bar, z, logit, logit_inv, logit_prime_inv, verbose=True)
    Q, state_list, index_of = build_transition_matrix(j0, xi_bar, z, f, f_inv, f_prime_inv, verbose=True)
    print(Q)

    print(f"State space size: {Q.shape}")


    print("Computing stationary distribution with GMRES...")
    tstart = time.perf_counter()
    # pi = stationary_distribution(Q, tol=1e-1)
    print(time.perf_counter() - tstart,'s elapsed')

    print("\nComputing stationary distribution with Arnoldi iteration...")
    tstart = time.perf_counter()
    pi_arnoldi = stationary_distribution_arnoldi(Q, k=1, verbose=True)
    print(time.perf_counter() - tstart,'s elapsed')

    # Compare the two methods
    # print(f"\nDifference between GMRES and Arnoldi: max|π_gmres - π_arnoldi| = {np.max(np.abs(pi - pi_arnoldi)):.2e}")

    # Use the Arnoldi result for the rest of the code
    pi = pi_arnoldi

    # Compute grid parameters consistently
    eta, epsilon, gamma, k_max = compute_grid_parameters(j0, xi_bar, f_inv, f_prime_inv)

    xs = np.array([j * eta for (j, k) in state_list])
    ys = np.array([k * gamma for (j, k) in state_list])

    # Grid dimensions
    j_max = j0
    n_j = 2 * j_max + 1
    n_k = 2 * k_max + 1
    N = n_j * n_k

    j_idx = np.repeat(np.arange(-j_max, j_max + 1), n_k)
    k_idx = np.tile(np.arange(-k_max, k_max + 1), n_j)

    x = j_idx * eta
    y = k_idx * gamma


    # compute marginals
    x_vals = np.unique(xs)
    y_vals = np.unique(ys)

    print(f"{x_vals=}")
    print(f"{y_vals=}")

    x_marg = np.zeros_like(x_vals)
    y_marg = np.zeros_like(y_vals)

    for val, prob in zip(xs, pi):
        x_marg[np.where(x_vals == val)[0][0]] += prob
    for val, prob in zip(ys, pi):
        y_marg[np.where(y_vals == val)[0][0]] += prob

    # Transform density back to original X space using binning
    print("\nTransforming density back to original space using binning...")
    # Use probability masses (not density) for binning
    X_vals, X_density = bin_density_to_original_space(x_vals, x_marg, f_inv, n_bins=100)

    # Compute statistics in original space
    # X_density is already a density, so compute probabilities from it
    x_bin_width = X_vals[1] - X_vals[0]  # uniform bin width
    X_marg_prob = X_density * x_bin_width
    X_marg_prob = X_marg_prob / np.sum(X_marg_prob)  # normalize to ensure it's a probability distribution
    X_mean = np.sum(X_vals * X_marg_prob)
    X_cumsum = np.cumsum(X_marg_prob)
    X_median = X_vals[np.searchsorted(X_cumsum, 0.5)]
    print(f"Original space: mean={X_mean:.4f}, median={X_median:.4f}")

    # comment out these lines if tex is not installed on your system
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.usetex'] = True

    # Compare transformed and original space densities
    plt.figure(figsize=(16,5))

    # Transformed space (ξ)
    plt.subplot(1,3,1)
    xi_mean = np.sum(x_vals * x_marg)
    xi_cumsum = np.cumsum(x_marg)
    xi_median = x_vals[np.searchsorted(xi_cumsum, 0.5)]
    plt.plot(x_vals, x_marg / eta)
    plt.axvline(xi_mean, color='red', linestyle='--', alpha=0.7, label=f'mean={xi_mean:.2f}')
    plt.axvline(xi_median, color='green', linestyle='--', alpha=0.7, label=f'median={xi_median:.2f}')
    plt.title(r"$\xi$-marginal of $\pi_\epsilon$ (transformed)")
    plt.xlabel(r"$\xi$")
    plt.ylabel("Density")
    plt.legend()

    # Original space (X)
    plt.subplot(1,3,2)
    plt.plot(X_vals, X_density)
    plt.axvline(X_mean, color='red', linestyle='--', alpha=0.7, label=f'mean={X_mean:.4f}')
    plt.axvline(X_median, color='green', linestyle='--', alpha=0.7, label=f'median={X_median:.4f}')
    plt.title(r"X-marginal of $\pi_\epsilon$ (original space)")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # marginal density for $B(1)$ (should match the standard normal)
    plt.subplot(1,3,3)
    y_mean = np.sum(y_vals * y_marg)
    y_cumsum = np.cumsum(y_marg)
    y_median = y_vals[np.searchsorted(y_cumsum, 0.5)]
    plt.plot(y_vals, y_marg / gamma)
    plt.axvline(y_mean, color='red', linestyle='--', alpha=0.7, label=f'mean={y_mean:.2f}')
    plt.axvline(y_median, color='green', linestyle='--', alpha=0.7, label=f'median={y_median:.2f}')
    plt.title(r"Y-marginal of $\pi_\epsilon$")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # Simulate and plot sample paths
    print("\nSimulating sample paths...")
    paths = simulate_paths(Q, state_list, n_paths=3, T_max=10.0, seed=42)
    fig = plot_sample_paths(paths, j0, xi_bar, z, f_inv, f_prime_inv,
                           title=f"Sample Paths (delta={delta}, z={z:.2f})")
    plt.savefig('sample_paths_transformed.png', dpi=150)
    plt.savefig('sample_paths_transformed.pdf')
    plt.show()

    # # grid of plots for different confidence levels
    # deltas = list(reversed([0.1, 0.2, 0.3, 0.45, 0.5, 0.55, 0.7, 0.8, 0.9]))

    # fig, axes = plt.subplots(3, 3, figsize=(8, 9))
    # axes = axes.flatten()

    # for ax, delta in zip(axes, deltas):
    #     z = z_from_delta(delta)
    #     Q, state_list, _ = build_transition_matrix(j0, xi_bar, z, f, f_inv, f_prime_inv)
    #     pi = stationary_distribution(Q)

    #     xs = np.array([j * eta for (j, k) in state_list])
    #     x_vals = np.unique(xs)
    #     x_marg_prob = np.array([pi[xs == xv].sum() for xv in x_vals])
    #     x_marg_density = x_marg_prob / eta

    #     x_mean = np.sum(x_vals * x_marg_prob)
    #     x_cumsum = np.cumsum(x_marg_prob)
    #     x_median = x_vals[np.searchsorted(x_cumsum, 0.5)]

    #     ax.plot(x_vals, x_marg_density)
    #     ax.axvline(x_mean, color='red', label=f'mean={x_mean:.2f}', linestyle='--', alpha=0.7, linewidth=1)
    #     ax.axvline(x_median, color='green', label=f'median={x_median:.2f}', linestyle='--', alpha=0.7, linewidth=1)
    #     ax.set_title(rf"$1-\eta={1-delta:.2f}$")
    #     ax.set_xlabel("x")
    #     ax.legend(fontsize=8, loc='best')

    # plt.tight_layout()
    # plt.savefig('grid_discretization.png')
    # plt.savefig('grid_discretization.pdf')
    # plt.show()