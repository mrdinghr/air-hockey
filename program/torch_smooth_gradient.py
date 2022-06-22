import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_air_hockey_baseline
from torch_EKF_Wrapper import air_hockey_EKF
device = torch.device("cuda")
table_length = 1.948


# input: recorded trajectories
# output:init_state of this trajectory
def calculate_init_state(data):
    state_dx = ((data[1][0] - data[0][0]) / (data[1][3] - data[0][3]) + (
            data[2][0] - data[1][0]) / (
                        data[2][3] - data[1][3]) + (data[3][0] - data[2][0]) / (
                        data[3][3] - data[2][3])) / 3
    state_dy = ((data[1][1] - data[0][1]) / (data[1][3] - data[0][3]) + (
            data[2][1] - data[1][1]) / (
                        data[2][3] - data[1][3]) + (data[3][1] - data[2][1]) / (
                        data[3][3] - data[2][3])) / 3
    state_dtheta = ((data[1][2] - data[0][2]) / (data[1][3] - data[0][3]) + (
            data[2][2] - data[1][2]) / (
                            data[2][3] - data[1][3]) + (data[3][2] - data[2][2]) / (
                            data[3][3] - data[2][3])) / 3
    state = torch.tensor([data[1][0], data[1][1], state_dx, state_dy, data[1][2], state_dtheta], device=device)
    return state


# input: trajectory, dyna_parameters, covariance_parameters
# output: state of one trajectory calculated by kalman smooth.
def state_kalman_smooth(cur_trajectory):

    return


class Kalman_Smooth_Gradient(torch.nn.Module):
    def __init__(self, params, covariance_params):
        super(Kalman_Smooth_Gradient, self).__init__()
        self.register_parameter('dyna_params', torch.nn.Parameter(params))
        self.covariance_params = covariance_params
        self.system = torch_air_hockey_baseline.SystemModel(tableDamping=self.dyna_params[1],
                                                            tableFriction=self.dyna_params[0],
                                                            tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                            puckRadius=0.03165, malletRadius=0.04815,
                                                            tableRes=self.dyna_params[3],
                                                            malletRes=0.8, rimFriction=self.dyna_params[2], dt=1 / 120)
        self.table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25,
                                                              puckRadius=0.03165,
                                                              restitution=self.dyna_params[2],
                                                              rimFriction=self.dyna_params[3], dt=1 / 120)
        self.R = torch.zeros((3, 3), device=device)
        self.R[0][0] = self.covariance_params[0]
        self.R[1][1] = self.covariance_params[1]
        self.R[2][2] = self.covariance_params[2]
        self.Q = torch.zeros((6, 6), device=device)
        self.Q[0][0] = self.covariance_params[3]
        self.Q[1][1] = self.covariance_params[3]
        self.Q[2][2] = self.covariance_params[4]
        self.Q[3][3] = self.covariance_params[4]
        self.Q[4][4] = self.covariance_params[5]
        self.Q[5][5] = self.covariance_params[6]
        self.P = torch.eye(6, device=device) * 0.01
        self.puck_EKF = air_hockey_EKF(u=1 / 120., system=self.system, table=self.table, Q=self.Q, R=self.R, P=self.P)

