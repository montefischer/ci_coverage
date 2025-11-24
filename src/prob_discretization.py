import time 

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import norm

def build_transition_matrix(n, z):
    """
    Build the sparse transition matrix P_epsilon for the Markov chain
    with sqrt(2*epsilon) = 1/n.  Reflection boundaries are used.

    Parameters
    ----------
    n : int
        Determines epsilon = 1/(2 n^2), grid spacings dx = 1/n^2, dy = 1/n.
    z : float
        Threshold defining the |y| <= z vs > z regions.

    Returns
    -------
    P : scipy.sparse.csr_matrix, shape (N, N)
        The transition matrix.
    state_list : list of (j,k)
        Mapping index -> state.
    index_of : dict mapping (j,k) -> index
    """

    eps = 1/(2*n*n)
    dy = 1/n

    # ---------------------------
    # Build state space
    # j runs from 0 to n^2/2, but 2 j eps <= 1 → 2 j/(2 n^2) <= 1 → j <= n^2
    # So j_max = n^2
    # k sqrt(2 eps) = k/n ∈ [ -sqrt(2/eps), sqrt(2/eps)] 
    # But sqrt(2/eps) = sqrt(4 n^2) = 2n
    # So k ∈ [-2n, 2n]
    # ---------------------------
    j_max = n*n  # since 2*j*eps = j/n^2 ≤ 1 → j ≤ n^2
    k_max = 6*n

    state_list = []
    index_of = {}

    for j in range(j_max + 1):
        for k in range(-k_max, k_max + 1):
            idx = len(state_list)
            state_list.append((j, k))
            index_of[(j, k)] = idx

    N = len(state_list)

    # Prepare arrays for CSR construction
    rows = []
    cols = []
    data = []

    # ---------------------------
    # Helper: add transition (i → j) with probability p
    # ---------------------------
    def add(i, j, p):
        rows.append(i)
        cols.append(j)
        data.append(p)

    # ---------------------------
    # Main construction loop
    # ---------------------------
    for idx, (j, k) in enumerate(state_list):

        # convert to y coordinate:
        y = k * dy

        # -------- Determine region ----------
        in_region = abs(y) <= z

        # -------- X transitions -------------
        if in_region:
            # inside: moves right
            p_right = (1 - 2*j*eps) / 2
            p_stay_x = j*eps
            # reflect if at max j
            if j < j_max:
                add(idx, index_of[(j+1, k)], p_right)
            else:
                # fully reflect probability back to self
                p_stay_x += p_right
        else:
            # outside: moves left
            p_left = j*eps
            p_stay_x = (1/2) - j*eps
            # reflect if at j=0
            if j > 0:
                add(idx, index_of[(j-1, k)], p_left)
            else:
                p_stay_x += p_left

        # -------- Y transitions -------------
        p_up =  1/4 - (k*eps)/4
        p_dn = 1/4 + (k*eps)/4

        # reflect in y if k hits ±k_max
        if k <  k_max:
            add(idx, index_of[(j, k+1)], p_up)
        else:
            p_stay_x += p_up

        if k > -k_max:
            add(idx, index_of[(j, k-1)], p_dn)
        else:
            p_stay_x += p_dn

        # -------- Add "stay" transition -----
        add(idx, idx, p_stay_x)

    # Build sparse matrix
    P = csr_matrix((data, (rows, cols)), shape=(N, N))
    return P, state_list, index_of

def stationary_distribution(P, tol=1e-12, max_iter=150_000):
    N = P.shape[0]
    v = np.ones(N) / N  # initial uniform distribution

    tstart = time.perf_counter()
    for iteration in range(max_iter):
        v_next = v @ P
        if np.linalg.norm(v_next - v, 1) < tol:
            print(f"Reached desired tolerance in {iteration} iterations, elapsed time {time.perf_counter() - tstart}")
            return v_next
        v = v_next
    print(f"Exceeded maximum iterations, elapsed time {time.perf_counter() - tstart}")
    return v  # return best estimate


def z_from_delta(delta):
    """Compute z such that P(-z <= N(0,1) <= z) = 1 - delta."""
    return norm.ppf(1 - delta / 2)


if __name__ == "__main__":
    # Configure matplotlib to use Computer Modern font and 12pt size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = 12
    plt.rcParams['text.usetex'] = True

    # Build and solve
    n = 50
    delta=0.8
    z = z_from_delta(delta)
    P, state_list, index_of = build_transition_matrix(n, z)
    print(f"{P.shape=}")
    pi = stationary_distribution(P)

    # Extract coordinates
    eps = 1/(2*n*n)
    dy = 1/n
    xs = np.array([2*j*eps for (j,k) in state_list])
    ys = np.array([k*dy for (j,k) in state_list])

    # Compute marginals
    x_vals = np.unique(xs)
    y_vals = np.unique(ys)

    x_marg = np.zeros_like(x_vals)
    y_marg = np.zeros_like(y_vals)

    for val, prob in zip(xs, pi):
        x_marg[np.where(x_vals == val)[0][0]] += prob
    for val, prob in zip(ys, pi):
        y_marg[np.where(y_vals == val)[0][0]] += prob

    # Plot marginals
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    dx = 2*eps
    x_mean = np.sum(x_vals * x_marg)
    x_cumsum = np.cumsum(x_marg)
    x_median = x_vals[np.searchsorted(x_cumsum, 0.5)]
    plt.plot(x_vals, x_marg / dx)
    plt.axvline(x_mean, color='red', linestyle='--', alpha=0.7, label=f'mean={x_mean:.2f}')
    plt.axvline(x_median, color='green', linestyle='--', alpha=0.7, label=f'median={x_median:.2f}')
    plt.title(r"X-marginal of $\pi$")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()

    plt.subplot(1,2,2)
    y_mean = np.sum(y_vals * y_marg)
    y_cumsum = np.cumsum(y_marg)
    y_median = y_vals[np.searchsorted(y_cumsum, 0.5)]
    plt.plot(y_vals, y_marg / dy)
    plt.axvline(y_mean, color='red', linestyle='--', alpha=0.7, label=f'mean={y_mean:.2f}')
    plt.axvline(y_median, color='green', linestyle='--', alpha=0.7, label=f'median={y_median:.2f}')
    plt.title(r"Y-marginal of $\pi$")
    plt.xlabel("y")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.show()


    deltas = list(reversed([0.1, 0.2, 0.3, 0.45, 0.5, 0.55, 0.7, 0.8, 0.9]))

    fig, axes = plt.subplots(3, 3, figsize=(7, 6))
    axes = axes.flatten()

    eps = 1/(2*n*n)

    for ax, delta in zip(axes, deltas):
        z = z_from_delta(delta)
        P, state_list, _ = build_transition_matrix(n, z)
        pi = stationary_distribution(P)

        xs = np.array([2*j*eps for (j,k) in state_list])
        x_vals = np.unique(xs)
        x_marg_prob = np.array([pi[xs == xv].sum() for xv in x_vals])
        x_marg = x_marg_prob / (2 * eps)

        x_mean = np.sum(x_vals * x_marg_prob)
        x_cumsum = np.cumsum(x_marg_prob)
        x_median = x_vals[np.searchsorted(x_cumsum, 0.5)]

        ax.plot(x_vals, x_marg)
        ax.axvline(x_mean, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x_median, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_title(rf"$1-\delta={1-delta:.2f}$")
        ax.set_xlabel("x")
        # ax.set_ylabel("density")
        ax.legend([f'mean={x_mean:.2f}', f'median={x_median:.2f}'], fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig('grid_discretization.png')
    plt.savefig('grid_discretization.pdf')
    plt.show()