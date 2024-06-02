import numpy as np

def pseudorange_model(x, SVs):
    return np.linalg.norm(x - SVs, axis=1)

def measurement_matrix(x, SVs):
    N = SVs.shape[0]
    rho = pseudorange_model(x, SVs)
    H = np.zeros((N, 3))
    for k in range(N):
      H[k, :] = (x - SVs[k, :]) / rho[k]
    return H

def solve_position(x_0, SVs, pranges, tol=1e-3, max_iters=100):
    H = measurement_matrix(x_0, SVs)
    z = pranges - pseudorange_model(x_0, SVs)
    delta_x = np.linalg.inv(H.T @ H) @ H.T @ z
    x_est = x_0 + delta_x

    iters = 1
    while np.linalg.norm(delta_x) >= tol and iters <= max_iters:
        H = measurement_matrix(x_est, SVs)
        z = pranges - pseudorange_model(x_est, SVs)
        delta_x = np.linalg.inv(H.T @ H) @ H.T @ z
        x_est += delta_x
        iters += 1
    return x_est