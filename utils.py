import numpy as np
from scipy.linalg import expm, logm

class Pose:
    def __init__(self, R, t):
        self.R = R
        self.t = t

class PoseGraph:
    def __init__(self):
        self.gnss_constraints = []
        self.odometry_constraints = []

    def add_gnss_constraint(self, i, satellite_positions, pseudoranges):
        self.gnss_constraints.append((i, satellite_positions, pseudoranges))
    
    def add_odometry_constraints(self, i, j, R, t):
        self.odometry_constraints.append((i, j, R, t))

def build_pose_graph(satellite_positions, pseudoranges, odoms):
    pose_graph = PoseGraph()

    # add GNSS constraints
    for i in range(len(satellite_positions)):
        pose_graph.add_gnss_constraint(i, satellite_positions[i], pseudoranges[i])

    # add odometry poses and constraints
    for i, (R, t) in enumerate(odoms):
        pose_graph.add_odometry_constraints(i, i + 1, R, t)

    return pose_graph

def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

def R_to_vec(R):
    log_R = logm(R)
    return [log_R[2, 1], log_R[0, 2], log_R[1, 0]]

def vec_to_R(v):
    return expm(skew_symmetric(v))

def theta_to_R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
def R_to_theta(R):
    return np.arctan2(R[1, 0], R[0, 0])