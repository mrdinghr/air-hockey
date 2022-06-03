import torch
import torch_air_hockey_baseline
from torch_EKF_Wrapper import air_hockey_EKF
from math import pi
import numpy as np
device = torch.device("cuda")


def calculate_loss(raw_data, params):
    table_friction = params[0]
    table_damping = params[1]
    table_restitution = params[2]
    system = torch_air_hockey_baseline.SystemModel(tableDamping=table_damping, tableFriction=table_friction,
                                                   tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                   puckRadius=0.03165, malletRadius=0.04815, tableRes=table_restitution,
                                                   malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
    table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                                     restitution=table_restitution, rimFriction=0.1418, dt=1 / 120)
    R = torch.zeros((3, 3), device=device)
    R[0][0] = 2.5e-7
    R[1][1] = 2.5e-7
    R[2][2] = 9.1e-3
    Q = torch.zeros((6, 6), device=device)
    Q[0][0] = Q[1][1] = 2e-10
    Q[2][2] = Q[3][3] = 3e-7
    Q[4][4] = 1.0e-2
    Q[5][5] = 1.0e-1
    P = torch.eye(6, device=device) * 0.01
    pre_data = []
    for j in range(1, len(raw_data)):
        if abs(raw_data[j][0] - raw_data[j - 1][0]) < 0.005 and abs(raw_data[j][1] - raw_data[j - 1][1]) < 0.005:
            continue
        pre_data.append(raw_data[j])
    pre_data = torch.tensor(pre_data)
    for m in pre_data:
        m[0] += table.m_length / 2
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
    state = torch.tensor([pre_data[1][0], pre_data[1][1], state_dx, state_dy, pre_data[1][2], state_dtheta], device=device)
    data = pre_data[1:]
    data = data.cuda()
    u = 1 / 120
    puck_EKF = air_hockey_EKF(state=state, u=u, system=system, table=table, Q=Q, R=R, P=P)
    num_evaluation = 0  # record the update time to normalize
    evaluation = 0  # calculate log_Ly_theta
    j = 1
    i = 0
    while j < len(data):
        i += 1
        if not puck_EKF.score:
            puck_EKF.predict()
            if (i - 0.2) / 120 < abs(data[j][-1] - data[0][-1]) < (i + 0.2) / 120:
                if abs(data[j - 1][2] - data[j][2]) > pi:
                    tmp = data[j][2]
                    data[j][2] += -torch.sign(data[j][2]) * pi + data[j - 1][2]
                    puck_EKF.update(data[j, 0:3])
                    data[j][2] = tmp
                else:
                    puck_EKF.update(data[j, 0:3])
                j += 1
                sign, logdet = torch.linalg.slogdet(puck_EKF.S)
                evaluation += (sign * torch.exp(logdet) + puck_EKF.y.T @ torch.linalg.inv(puck_EKF.S) @ puck_EKF.y)
                num_evaluation += 1

            else:
                if abs(data[j][-1] - data[0][-1]) <= (i - 0.2) / 120:
                    j += 1
                puck_EKF.state = puck_EKF.predict_state
        else:
            if abs(data[j - 1][2] - data[j - 2][2]) > pi:
                rotation_velocity = (data[j - 1][2] - torch.sign(data[j - 1][2]) * pi) / (
                        data[j - 1][-1] - data[j - 2][-1])
            else:
                rotation_velocity = (data[j - 1][2] - data[j - 2][2]) / (data[j - 1][3] - data[j - 2][3])
            puck_EKF.state = torch.tensor(
                [data[j - 1][0], data[j - 1][1],
                 (data[j - 1][0] - data[j - 2][0]) / (data[j - 1][3] - data[j - 2][3]),
                 (data[j - 1][1] - data[j - 2][1]) / (data[j - 1][3] - data[j - 2][3]), data[j - 1][2],
                 rotation_velocity], dtype=float, device=device)
            puck_EKF.predict()
            j += 1
    return evaluation / num_evaluation


class EKFGradient(torch.nn.Module):
    def __init__(self, raw_data):
        super(EKFGradient, self).__init__()
        self.raw_data = raw_data

    def forward(self, params):
        return calculate_loss(self.raw_data, params)


raw_data = np.load("example_data2.npy")
model = EKFGradient(raw_data)
model.to(device)
init_params = torch.tensor([0.5, 0.5, 0.5])
print(model.forward(init_params))
learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for t in range(5):
#     loss = model.forward(init_params)
#     print(t, loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
