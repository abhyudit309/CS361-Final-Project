import numpy as np
from scipy.optimize import least_squares
from utils import Pose, PoseGraph, build_pose_graph, R_to_vec, vec_to_R, theta_to_R, R_to_theta
from tqdm import tqdm

## 3D residuals
def gnss_residual_3D(pose_graph: PoseGraph, params, sigma_gps):
    residuals = []
    for (i, satellite_positions, pseudoranges) in pose_graph.gnss_constraints:
        ti = params[6 * i + 3 : 6 * i + 6]
        for j in range(len(pseudoranges)):
            sat_pos = satellite_positions[j]
            prange = pseudoranges[j]
            pred_range = np.linalg.norm(ti - sat_pos)

            # range residual
            range_residual = (pred_range - prange) / sigma_gps
            residuals.append(range_residual)

    return np.array(residuals)

def odometry_residual_3D(pose_graph: PoseGraph, params, sigma_R, sigma_t):
    residuals = []
    for (i, j, R, t) in pose_graph.odometry_constraints:
        Ri = vec_to_R(params[6 * i : 6 * i + 3])
        ti = params[6 * i + 3 : 6 * i + 6]
        Rj = vec_to_R(params[6 * j : 6 * j + 3])
        tj = params[6 * j + 3 : 6 * j + 6]

        # odometry residuals
        R_res = (Ri @ R - Rj) / sigma_R
        t_res = (Ri @ t - (tj - ti)) / sigma_t

        residuals.extend(R_res.flatten())
        residuals.extend(t_res)
    
    return np.array(residuals)

def combined_residuals_3D(pose_graph: PoseGraph, params, sigma_gps, sigma_R, sigma_t):
    gnss_res = gnss_residual_3D(pose_graph, params, sigma_gps)
    odom_res = odometry_residual_3D(pose_graph, params, sigma_R, sigma_t)
    return np.hstack((gnss_res, odom_res))

## 2D residuals
def gnss_residual_2D(pose_graph: PoseGraph, params, sigma_gps):
    residuals = []
    for (i, satellite_positions, pseudoranges) in pose_graph.gnss_constraints:
        ti = np.append(params[3 * i + 1 : 3 * i + 3], 0.0)
        for j in range(len(pseudoranges)):
            sat_pos = satellite_positions[j]
            prange = pseudoranges[j]
            pred_range = np.linalg.norm(ti - sat_pos)

            # range residual
            range_residual = (pred_range - prange) / sigma_gps
            residuals.append(range_residual)

    return np.array(residuals)

def odometry_residual_2D(pose_graph: PoseGraph, params, sigma_R, sigma_t):
    residuals = []
    for (i, j, R, t) in pose_graph.odometry_constraints:
        Ri = theta_to_R(params[3 * i])
        ti = params[3 * i + 1 : 3 * i + 3]
        Rj = theta_to_R(params[3 * j])
        tj = params[3 * j + 1 : 3 * j + 3]

        # odometry residuals
        R_res = (Ri @ R - Rj) / sigma_R
        t_res = (Ri @ t - (tj - ti)) / sigma_t

        residuals.extend(R_res.flatten())
        residuals.extend(t_res)
    
    return np.array(residuals)

def combined_residuals_2D(pose_graph: PoseGraph, params, sigma_gps, sigma_R, sigma_t):
    gnss_res = gnss_residual_2D(pose_graph, params, sigma_gps)
    odom_res = odometry_residual_2D(pose_graph, params, sigma_R, sigma_t)
    return np.hstack((gnss_res, odom_res))

def optimize_pose_graph_window(pose_graph: PoseGraph, initial_poses, sigma_gps, sigma_R, sigma_t, optimize_3D):
    # unpack initial conditions
    initial_conditions = []
    for pose in initial_poses:
        if optimize_3D:
            initial_conditions.extend(R_to_vec(pose.R))
        else:
            initial_conditions.append(R_to_theta(pose.R))
        initial_conditions.extend(pose.t)
    initial_conditions = np.array(initial_conditions)

    # use a lambda
    if optimize_3D:
        residuals_func = lambda params: combined_residuals_3D(pose_graph, params, sigma_gps, sigma_R, sigma_t)
        ftol = 1.0
    else:
        residuals_func = lambda params: combined_residuals_2D(pose_graph, params, sigma_gps, sigma_R, sigma_t)
        ftol = 0.1
    result = least_squares(residuals_func, initial_conditions, max_nfev=10, ftol=ftol)

    optimized_params = result.x
    optimized_poses = []
    
    for i in range(len(initial_poses)):
        if optimize_3D:
            R = vec_to_R(optimized_params[6 * i : 6 * i + 3])
            t = optimized_params[6 * i + 3 : 6 * i + 6]
        else:
            R = theta_to_R(optimized_params[3 * i])
            t = optimized_params[3 * i + 1 : 3 * i + 3]
        optimized_poses.append(Pose(R, t))

    return optimized_poses

def build_and_optimize_pose_graph(satellite_positions, pseudoranges, odoms, window_size, shift, sigma_gps, sigma_R, sigma_t, optimize_3D):
    N = len(pseudoranges) # length of trajectory
    W = window_size
    S = shift
    n = 3 if optimize_3D else 2

    # Maximum start index for sliding window
    k_max = (N - W) // S
    graph_positions = np.zeros((N, n))

    # initial conditions
    initial_poses = []
    initial_pose = Pose(np.eye(n), np.zeros(n))
    initial_poses.append(initial_pose)
    for i in range(W - 1):
        current_pose = initial_poses[i]
        Rc, tc = current_pose.R, current_pose.t
        # use odometry to initialize poses
        Rn = Rc @ odoms[i][0]
        tn = Rc @ odoms[i][1] + tc
        next_pose = Pose(Rn, tn)
        initial_poses.append(next_pose)

    # For loop for sliding window optimization
    for k in (pbar := tqdm(range(k_max + 1))):
        # Get indices for current window
        window = slice(k * S, k * S + W)

        # Extract odometry, ranges, and satellite positions for window
        satellite_positions_window = satellite_positions[window]
        pranges_window = pseudoranges[window]
        odoms_window = odoms[window][:-1] # discard the last one

        # build the graph and optimize
        pose_graph = build_pose_graph(satellite_positions_window, pranges_window, odoms_window)
        optimized_poses = optimize_pose_graph_window(pose_graph, initial_poses, sigma_gps, sigma_R, sigma_t, optimize_3D)

        # Extract and save positions from window
        graph_positions[window] = [pose.t for pose in optimized_poses]

        # Update initial poses and use odometry to initialize latest new poses shifted into the window
        if k < k_max:
            initial_poses[: W - S] = optimized_poses[S:]
            idx = slice(k * S + W - 1, (k + 1) * S + W - 1)
            next_odoms = odoms[idx]
            for i in range(S):
                R_odom, t_odom = next_odoms[i]
                pose = initial_poses[W - S + i - 1]
                R, t = pose.R, pose.t
                Rn = R @ R_odom
                tn = R @ t_odom + t
                initial_poses[W - S + i] = Pose(Rn, tn)
    # end of optimization
    return graph_positions