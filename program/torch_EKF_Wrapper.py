from math import pi
import torch
import torch_air_hockey_baseline
from matplotlib import pyplot as plt
import numpy as np
device = torch.device("cuda")


class air_hockey_EKF:
    def __init__(self, u, system, table, Q, R, P):
        self.state = None
        self.system = system
        self.table = table
        self.Q = Q
        self.R = R
        self.P = P
        self.u = u
        self.predict_state = None
        self.F = None
        self.score = False
        self.has_collision = False
        self.H = torch.zeros((3, 6), device=device)
        self.H[0][0] = self.H[1][1] = self.H[2][4] = 1
        self.y = None
        self.S = None

    def init_state(self, state):
        self.state = state

    def predict(self):
        self.P = self.system.F @ self.P @ self.system.F.T + self.Q
        self.has_collision, self.predict_state, jacobian, self.score = self.table.apply_collision(self.state)
        if self.has_collision:
            self.F = jacobian
        else:
            self.F = self.system.F
            self.predict_state = self.system.f(self.state, self.u)

    def update(self, measure):
        # measurement residual
        self.y = measure - self.predict_state[[0, 1, 4]]
        if self.y[2] >= pi:
            self.y[2] -= pi * 2
        elif self.y[2] <= -pi:
            self.y[2] += pi * 2
        self.S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(self.S)
        self.state = self.predict_state + K @ self.y
        self.P = (torch.eye(6, device=device) - K @ self.H) @ self.P



# test for torch_EKF_Wrapper
# tableDamping = 0.001
# tableFriction = 0.001
# tableRestitution = 0.7424
system = torch_air_hockey_baseline.SystemModel(tableDamping=0.001, tableFriction=0.001, tableLength=1.948, tableWidth=1.038,
                                         goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
                                         tableRes=0.7424, malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                           restitution=0.7424, rimFriction=0.1418, dt=1 / 120)
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
pre_data = np.load("example_data.npy")
data = []
for i in range(1, len(pre_data)):
    if abs(pre_data[i][0] - pre_data[i - 1][0]) < 0.005 and abs(pre_data[i][1] - pre_data[i - 1][1]) < 0.005:
        continue
    data.append(pre_data[i])
for i_data in data:
    i_data[0] += table.m_length / 2
data = torch.tensor(data, device=device).float()
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
puck_EKF = air_hockey_EKF(u=1 / 120, system=system, table=table, Q=Q, R=R, P=P)
puck_EKF.init_state(state)
resx = [state[0]]
resy = [state[1]]
res_theta = [state[4]]
time_EKF = [1/120]
j = 1
length = len(data)-1
i = 0
while j < length:
    i += 1
    time_EKF.append((i + 1) / 120)
    puck_EKF.predict()
    resx.append(puck_EKF.predict_state[0])
    resy.append(puck_EKF.predict_state[1])
    res_theta.append(puck_EKF.predict_state[4])
    # check whether data is recorded at right time
    if (i-0.2) / 120 < data[j+1][-1]-data[1][-1] < (i+0.2) / 120:
        puck_EKF.update(data[j + 1][0:3])
        j += 1
    elif data[j+1][-1]-data[1][-1] <= (i-0.2) / 120:
        j += 1
        puck_EKF.state = puck_EKF.predict_state
    else:
        puck_EKF.state = puck_EKF.predict_state
    # else:
        # if torch.abs(data[j - 1][2] - data[j][2]) > pi:
        #     rotation_velocity = (data[j][2] - torch.sign(data[j][2])*pi) / (data[j][-1] - data[j - 1][-1])
        # else:
        #     rotation_velocity = (data[j - 1][2] - data[j][2]) / (data[j - 1][3] - data[j][3])
        # puck_EKF.state = torch.tensor(
        #     [data[j][0], data[j][1], (data[j - 1][0] - data[j][0]) / (data[j - 1][3] - data[j][3]),
        #      (data[j - 1][1] - data[j][1]) / (data[j - 1][3] - data[j][3]), data[j][2], rotation_velocity], device=device)
        # puck_EKF.predict()
        # resx.append(puck_EKF.predict_state[0])
        # resy.append(puck_EKF.predict_state[1])
        # res_theta.append(puck_EKF.predict_state[4])
        # j += 1
resx = torch.tensor(resx, device=device)
resy = torch.tensor(resy, device=device)
res_theta = torch.tensor(res_theta, device=device)
plt.figure()
plt.scatter(data[1:, 0].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g', label='raw data', s=5)
plt.scatter(resx.cpu().numpy(), resy.cpu().numpy(), color='b', label='EKF', s=5)
plt.legend()
plt.figure()
plt.subplot(3, 3, 1)
plt.scatter(time_EKF, resx.cpu().numpy(), color='b', label='EKF x position', s=5)
plt.title('only EKF x position')
plt.legend()
plt.subplot(3, 3, 2)
plt.scatter(data[1:, -1].cpu().numpy()-data[0][-1].cpu().numpy(), data[1:, 0].cpu().numpy(), color='g', label='raw data x position', s=5)
plt.title('only raw data x position')
plt.legend()
plt.subplot(3, 3, 3)
plt.scatter(time_EKF, resx.cpu().numpy(), color='b', label='EKF x position', s=5)
plt.scatter(data[1:, -1].cpu().numpy()-data[0][-1].cpu().numpy(), data[1:, 0].cpu().numpy(), color='g', label='raw data x position', s=5)
plt.title('EKF vs raw data x position')
plt.legend()
plt.subplot(3, 3, 4)
plt.scatter(time_EKF, resy.cpu().numpy(), color='b', label='EKF y position', s=5)
plt.title('only EKF y position')
plt.legend()
plt.subplot(3, 3, 5)
plt.scatter(data[1:, -1].cpu().numpy()-data[0][-1].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g', label='raw data y position', s=5)
plt.title('only raw data y position')
plt.legend()
plt.subplot(3, 3, 6)
plt.scatter(time_EKF, resy.cpu().numpy(), color='b', label='EKF y position', s=5)
plt.scatter(data[1:, -1].cpu().numpy()-data[0][-1].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g', label='raw data y position', s=5)
plt.title('EKF vs raw data y position')
plt.legend()
plt.subplot(3, 3, 7)
plt.scatter(time_EKF, res_theta.cpu().numpy(), color='b', label='EKF theta', s=5)
plt.title('only EKF theta')
plt.legend()
plt.subplot(3, 3, 8)
plt.scatter(data[1:, -1].cpu().numpy()-data[0][-1].cpu().numpy(), data[1:, 2].cpu().numpy(), color='g', label='raw data theta', s=5)
plt.legend()
plt.subplot(3, 3, 9)
plt.scatter(time_EKF,  res_theta.cpu().numpy(), color='b', label='EKF theta', s=5)
plt.scatter(data[1:, -1].cpu().numpy()-data[0][-1].cpu().numpy(), data[1:, 2].cpu().numpy(), color='g', label='raw data y position', s=5)
plt.legend()
plt.show()

