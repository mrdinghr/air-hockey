import torch
import torch_air_hockey_baseline
from torch_EKF_Wrapper import air_hockey_EKF
from math import pi
import numpy as np
device = torch.device("cuda")


def preprocess_data(raw_data, model):
    pre_data = []
    for j in range(1, len(raw_data)):
        if abs(raw_data[j][0] - raw_data[j - 1][0]) < 0.005 and abs(raw_data[j][1] - raw_data[j - 1][1]) < 0.005:
            continue
        pre_data.append(raw_data[j])
    pre_data = torch.tensor(pre_data, device=device)
    for m in pre_data:
        m[0] = m[0] + model.system.tableLength / 2
    # EKF initialized state
    state_dx = ((pre_data[1][0] - pre_data[0][0]) / (pre_data[1][3] - pre_data[0][3]) + (
            pre_data[2][0] - pre_data[1][0]) / (
                        pre_data[2][3] - pre_data[1][3]) + (pre_data[3][0] - pre_data[2][0]) / (
                        pre_data[3][3] - pre_data[2][3])) / 3
    state_dy = ((pre_data[1][1] - pre_data[0][1]) / (pre_data[1][3] - pre_data[0][3]) + (
            pre_data[2][1] - pre_data[1][1]) / (
                        pre_data[2][3] - pre_data[1][3]) + (pre_data[3][1] - pre_data[2][1]) / (
                        pre_data[3][3] - pre_data[2][3])) / 3
    state_dtheta = ((pre_data[1][2] - pre_data[0][2]) / (pre_data[1][3] - pre_data[0][3]) + (
            pre_data[2][2] - pre_data[1][2]) / (
                            pre_data[2][3] - pre_data[1][3]) + (pre_data[3][2] - pre_data[2][2]) / (
                            pre_data[3][3] - pre_data[2][3])) / 3
    state = torch.tensor([pre_data[1][0], pre_data[1][1], state_dx, state_dy, pre_data[1][2], state_dtheta],
                         device=device)
    pre_data = pre_data.float()
    return pre_data, state


class EKFGradient(torch.nn.Module):
    def __init__(self, params, covariance_params):
        super(EKFGradient, self).__init__()
        self.register_parameter('dyna_params', torch.nn.Parameter(params))
        self.covariance_params = covariance_params
        # self.register_parameter('covariance_params', covariance_params)

        self.system = torch_air_hockey_baseline.SystemModel(tableDamping=self.dyna_params[1], tableFriction=self.dyna_params[0],
                                                       tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                       puckRadius=0.03165, malletRadius=0.04815, tableRes=self.dyna_params[2],
                                                       malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
        self.table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                                         restitution=self.dyna_params[2], rimFriction=0.1418, dt=1 / 120)

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

    def forward(self, raw_data):
        loss = self.calculate_loss(raw_data)
        return loss

    def calculate_loss(self, raw_data, state):
        self.puck_EKF.init_state(state)
        data = raw_data[1:]
        num_evaluation = 0  # record the update time to normalize
        evaluation = torch.tensor([0], device=device, dtype=float)
        # evaluation = torch.zeros(len(data)-1, dtype=float, device=device)   # calculate log_Ly_theta
        # evaluation = torch.tensor(0., device=device)
        j = 1
        i = 0
        while j < len(data) - 1:
            i += 1
            self.puck_EKF.predict()
            # check whether data is recorded at right time
            if (i - 0.2) / 120 < data[j][-1] - data[0][-1] < (i + 0.2) / 120:
                self.puck_EKF.update(data[j + 1][0:3])
                j += 1
                sign, logdet = torch.linalg.slogdet(self.puck_EKF.S)
                # evaluation[j-2] = evaluation[j-2] + sign * torch.exp(logdet) + self.puck_EKF.y.T @ torch.linalg.inv(self.puck_EKF.S) @ self.puck_EKF.y
                evaluation =torch.cat((evaluation, torch.tensor([sign * torch.exp(logdet) + self.puck_EKF.y.T @ torch.linalg.inv(self.puck_EKF.S) @ self.puck_EKF.y], device=device)))
                # evaluation += sign * torch.exp(logdet) + self.puck_EKF.y.T @ torch.linalg.inv(self.puck_EKF.S) @ self.puck_EKF.y
                num_evaluation += 1
            elif data[j + 1][-1] - data[1][-1] <= (i - 0.2) / 120:
                j += 1
                self.puck_EKF.state = self.puck_EKF.predict_state
            else:
                self.puck_EKF.state = self.puck_EKF.predict_state
        # evaluation.requires_grad
        # evaluation.retain_grad()
        loss = torch.sum(evaluation) / num_evaluation
        loss.requires_grad_(True)
        # loss = evaluation / num_evaluation
        return loss


init_params = torch.Tensor([0.125, 0.375, 0.6749999523162842])
covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 3e-7, 1.0e-2, 1.0e-1])
model = EKFGradient(init_params, covariance_params)

raw_data = np.load("example_data2.npy")
raw_data, init_state = preprocess_data(raw_data, model)

# model.to(device)
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# use autograd to optimize parameters
for t in range(10):
    loss = model.calculate_loss(raw_data, init_state)
    print(t, loss)
    print(model.get_parameter('dyna_params'))
    # print(model.get_parameter('covparams'))
    optimizer.zero_grad()
    loss.backward()
    print(init_params.grad)
    # print(covariance_params.grad)
    optimizer.step()
    for p in model.get_parameter('dyna_params'):
        p.data.clamp_(0, 1)
    # for p in model.get_parameter('covparams'):
    #     p.data.clamp_(-100, 100)
