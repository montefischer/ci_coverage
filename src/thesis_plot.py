"""
Density of Z(1) for several confidence levels 1 - delta, from the stationary
distribution of the approximating Markov chain (Figure "grid" in the paper).
"""
import argparse
import os
import time
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigs, splu
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


def build_folded_transition_matrix(j_0, z, verbose=True):
    """
    P_epsilon lumped onto (j, |k|). The chain is symmetric under k -> -k, so this
    gives the same x-marginal on half the states.
    """
    t0 = time.perf_counter()
    eta = 1 / j_0
    gamma = 1 / np.sqrt(j_0)
    k_max = int(np.ceil(6 * np.sqrt(j_0)))
    n_m = k_max + 1
    n_j = j_0 + 1
    N = n_j * n_m

    j_idx = np.repeat(np.arange(n_j), n_m)
    m_idx = np.tile(np.arange(n_m), n_j)
    in_region = m_idx * gamma <= z

    p_right = np.where(in_region & (j_idx < j_0), (1 - j_idx * eta) / 2, 0.0)
    p_left = np.where(~in_region & (j_idx > 0), j_idx * eta / 2, 0.0)
    p_up = np.where(m_idx == 0, 0.5, 0.25 - (m_idx * eta) / 8)
    p_up = np.where(m_idx == k_max, 0.0, p_up)
    p_dn = np.where(m_idx == 0, 0.0, 0.25 + (m_idx * eta) / 8)
    p_stay = 1 - p_right - p_left - p_up - p_dn

    P = diags([p_stay, p_left[n_m:], p_right[:-n_m], p_dn[1:], p_up[:-1]],
              [0, -n_m, n_m, -1, 1], shape=(N, N), format="csr")
    if verbose:
        print(f"Folded matrix ({N} states) built in {time.perf_counter() - t0:.4f}s")
    return P


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


def stationary_distribution_lu(P, verbose=True):
    # sparse LU solve of pi (P - I) = 0 with pi[s0] = 1 pinned, then normalize
    tstart = time.perf_counter()
    n = P.shape[0]
    A = (P.T - identity(n, format="csr")).tocsc()
    s0 = n // 2
    keep = np.ones(n, bool); keep[s0] = False
    lu = splu(A[keep][:, keep].tocsc(), permc_spec="MMD_AT_PLUS_A")
    rest = lu.solve(-A[keep][:, [s0]].toarray().ravel())
    pi = np.empty(n); pi[keep] = rest; pi[s0] = 1.0
    if verbose:
        print(f"Stationary distribution found in {time.perf_counter() - tstart:.2f}s")
    return pi / pi.sum()


def z_from_delta(delta):
    """Return z such that P(-z <= N(0,1) <= z) = 1 - delta."""
    return norm.ppf(1 - delta / 2)


def get_stats(support, density):
    mean = np.sum(support * density)
    cumsum = np.cumsum(density)
    median = support[np.searchsorted(cumsum, 0.5)]

    return mean, median


def load_cache(path, key):
    if not os.path.exists(path):
        return None
    try:
        print(f"loading {path}")
        return np.load(path)[key]
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as e:
        print(f"could not read {path} ({e}); recomputing")
        return None


def x_marginal(j_0, delta, cache_dir, recompute=False, solver='lu'):
    z = z_from_delta(delta)
    xvals = np.arange(j_0 + 1) / j_0
    if solver == 'lu':
        path = os.path.join(cache_dir, f'folded{j_0}_delta{delta}.npz')
        xmarg = None if recompute else load_cache(path, 'xmarg')
        if xmarg is None:
            P = build_folded_transition_matrix(j_0, z)
            xmarg = stationary_distribution_lu(P).reshape(j_0 + 1, -1).sum(axis=1)
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(path, j_0=j_0, delta=delta, z=z, xmarg=xmarg)
        return xvals, xmarg
    path = os.path.join(cache_dir, f'coarse{j_0}_delta{delta}.npz')
    pi = None if recompute else load_cache(path, 'pi')
    if pi is None:
        P, state_list, index_of = build_transition_matrix(j_0, z)
        pi = stationary_distribution(P)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, j_0=j_0, delta=delta, z=z, pi=pi)
    return xvals, pi.reshape(j_0 + 1, -1).sum(axis=1)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--j0', type=int, default=10_000, help='grid resolution: eta = 1/j0, gamma = 1/sqrt(j0)')
    parser.add_argument('--cache-dir', default=os.path.join(here, '..', 'discretizations'),
                        help='directory of cached stationary distributions')
    parser.add_argument('--recompute', action='store_true', help='ignore cached stationary distributions')
    parser.add_argument('--solver', choices=['lu', 'arnoldi'], default='lu',
                        help='lu: lumped chain, sparse LU; arnoldi: full chain, eigs (slow)')
    parser.add_argument('--logy', action='store_true', help='plot densities on a log y scale')
    parser.add_argument('--no-truncate', action='store_true',
                        help='linear scale: show endpoint spikes in full instead of truncating them')
    parser.add_argument('--out', default=None,
                        help='output figure path (default grid_discretization_thesis[_logy].pdf)')
    args = parser.parse_args()
    if args.out is None:
        args.out = 'grid_discretization_thesis_logy.pdf' if args.logy else 'grid_discretization_thesis.pdf'

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.usetex'] = True
    # printed size: \linewidth = 380pt in apt.cls
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    LINEWIDTH_IN = 380 / 72.27

    j_0 = args.j0
    print(f"Grid parameters: j_0 = {j_0}, eta = {1/j_0:.6f}, gamma = {1/np.sqrt(j_0):.6f}")

    fig, axes = plt.subplots(3, 3, figsize=(LINEWIDTH_IN, LINEWIDTH_IN * 9 / 8))
    axes = axes.flatten()
    conf_levels = list(reversed([0.05, 0.1, 0.2, 0.3, 0.45, 0.5, 0.55, 0.7, 0.9]))

    for ax, conf_level in zip(axes, conf_levels):
        z = z_from_delta(conf_level)
        print(f"Confidence level 1-delta = {1-conf_level:.2f}, z = {z:.4f}")
        xvals, xmarg = x_marginal(j_0, conf_level, args.cache_dir, args.recompute, args.solver)
        mean, med = get_stats(xvals, xmarg)

        density = xmarg * j_0
        line, = ax.plot(xvals, density)
        Y_CAP = {0.45: 2, 0.50: 2, 0.55: 2}.get(round(1 - conf_level, 2), 10)
        if not args.logy and not args.no_truncate and ax.get_ylim()[1] > Y_CAP:
            ax.set_ylim(-0.04 * Y_CAP, Y_CAP)
        if not args.logy:
            if density[0] > density[-1]:
                strip = xvals <= 0.01
                mass_label = rf'$P(Z(1){{<}}0.01){{=}}{xmarg[xvals < 0.01].sum():.2f}$'
            else:
                strip = xvals >= 0.99
                mass_label = rf'$P(Z(1){{>}}0.99){{=}}{xmarg[xvals > 0.99].sum():.2f}$'
            ax.fill_between(xvals[strip], 0, density[strip], color=line.get_color(), alpha=0.3,
                            linewidth=0, label=mass_label)
        if args.logy:
            ax.set_yscale('log')
            lo, hi = ax.get_ylim()
            if lo < 1e-4:
                peak = density.max()
                hi = peak * (peak / 1e-4) ** 0.05
            ax.set_ylim(max(lo, 1e-4), min(hi, 1e4))
        med_str = f'{med:.2f}' if f'{med:.2f}' != '1.00' else f'{med:.3f}'
        ax.axvline(mean, color='red', linestyle='--', alpha=0.7, linewidth=1, label=f'mean={mean:.2f}')
        ax.axvline(med, color='green', linestyle=':', alpha=0.9, linewidth=1.3, label=f'median={med_str}')
        ax.grid(True, color='0.85', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(rf"$1-\delta={1-conf_level:.2f}$")
        ax.set_xlabel("x")
        ax.legend(fontsize=7, loc='best', handlelength=1.2, handletextpad=0.4, borderpad=0.3,
                  labelspacing=0.25, borderaxespad=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"saved {args.out}")
