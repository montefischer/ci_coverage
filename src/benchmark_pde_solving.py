import os
MAX_THREADS = 8

os.environ["OMP_NUM_THREADS"] = f"{MAX_THREADS}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{MAX_THREADS}"
os.environ["MKL_NUM_THREADS"] = f"{MAX_THREADS}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{MAX_THREADS}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{MAX_THREADS}"


import time
import os
from pde_markov_discretization import *

if __name__ == '__main__':
    os.makedirs('density', exist_ok=True)
    xres_array = [100, 200, 300, 400, 500, 1000, 2500, 5000, 10000]
    for j_0 in xres_array:
        delta = 0.5
        z = z_from_delta(delta)

        print(f"Grid parameters: j_0 = {j_0}, eta = {1/j_0:.6f}, gamma = {1/np.sqrt(j_0):.6f}")
        print(f"Confidence level 1-delta = {1-delta:.2f}, z = {z:.4f}")

        P, state_list, index_of = build_transition_matrix(j_0, z)
        print(f"State space size: {P.shape}")
        tstart = time.perf_counter()
        pi = stationary_distribution(P)
        tend = time.perf_counter()
        print(f"Solved for stationary distribution in {tend - tstart:.1f} s")
        np.savez(f'density/coarse{j_0}_delta{delta}.npz', j_0=j_0, delta=delta, z=z, pi=pi)
        mean, median = get_stats(*get_x_density(j_0, state_list, pi))
        print(f"{mean=}")
        print(f"{median=}")
    
        print('---\n')
