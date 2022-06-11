import torch
import torch_air_hockey_baseline
from torch_EKF_Wrapper import air_hockey_EKF
from matplotlib import pyplot as plt
from math import pi
import numpy as np
device = torch.device("cuda")
table_length = 1.948


def preprocess_data(pre_data):
    data = []
    for i in range(1, len(pre_data)):
        if abs(pre_data[i][0] - pre_data[i - 1][0]) < 0.005 and abs(pre_data[i][1] - pre_data[i - 1][1]) < 0.005:
            continue
        data.append(pre_data[i])
    for i_data in data:
        i_data[0] += table_length / 2
    data = torch.tensor(data, device=device).float()
    # EKF initialized state
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
    return data, state


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
        data = raw_data
        evaluation = 0
        num_evaluation = 0  # record the update time to normalize
        # evaluation = torch.tensor([0], device=device, dtype=float, requires_grad=True)
        # evaluation = torch.zeros(len(data)-2, dtype=float, device=device)   # calculate log_Ly_theta
        # evaluation = 0
        j = 1
        i = 0
        while j < len(data) - 1:
            i = i + 1
            self.puck_EKF.predict()
            # check whether data is recorded at right time
            if (i - 0.2) / 120 < data[j+1][-1] - data[1][-1] < (i + 0.2) / 120:
                self.puck_EKF.update(data[j + 1][0:3])
                j = j + 1
                sign, logdet = torch.linalg.slogdet(self.puck_EKF.S)
                # cur_log = sign * torch.exp(logdet) + self.puck_EKF.y.T @ torch.linalg.inv(self.puck_EKF.S) @ self.puck_EKF.y
                # evaluation[j-2] = cur_log
                # evaluation =torch.cat((evaluation, torch.tensor([cur_log], device=device)))
                evaluation = evaluation + sign * torch.exp(logdet) + self.puck_EKF.y.T @ torch.linalg.inv(self.puck_EKF.S) @ self.puck_EKF.y
                num_evaluation += 1
            elif data[j + 1][-1] - data[1][-1] <= (i - 0.2) / 120:
                j = j + 1
                self.puck_EKF.state = self.puck_EKF.predict_state
            else:
                self.puck_EKF.state = self.puck_EKF.predict_state
        cur_loss = evaluation / num_evaluation
        # evaluation.requires_grad
        # evaluation.retain_grad()
        # loss = torch.mean(evaluation)
        # loss = evaluation / num_evaluation
        # cur_loss.requires_grad_(True)
        return cur_loss


init_params = torch.Tensor([0.125, 0.375, 0.6749999523162842])
covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1])
model = EKFGradient(init_params, covariance_params)
pre_data = np.load("example_data2.npy")
raw_data, init_state = preprocess_data(pre_data)
# model.to(device)
learning_rate = 1e-7
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# use autograd to optimize parameters
for t in range(50):
    loss = model.calculate_loss(raw_data, init_state)
    plt.scatter(t, loss.item(), color='b')
    print(t, loss)
    print('dyna_params:')
    print(model.get_parameter('dyna_params'))
    # print(model.get_parameter('covparams'))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    print('grad:')
    print(model.get_parameter('dyna_params').grad)
    # print(covariance_params.grad)
    optimizer.step()
    for p in model.get_parameter('dyna_params'):
        p.data.clamp_(0, 1)
    # for p in model.get_parameter('covparams'):
    #     p.data.clamp_(-100, 100)
plt.title('loss curve')
plt.show()
