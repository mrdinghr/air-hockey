import numpy.linalg as lg
import numpy as np
import air_hockey_baseline
import matplotlib.pyplot as plt
from air_hockey_plot import table_plot
from math import pi

class air_hockey_EKF:
    def __init__(self, state, u, system, table, Q, R, P):
        self.state = state
        self.system = system
        self.table = table
        self.Q = Q
        self.R = R
        self.P = P
        self.u = u
        self.predict_state = None
        self.F = None
        self.score = False
        self.y = None
        self.S = None

    def predict(self):
        self.P = self.system.F @ self.P @ self.system.F.T + self.Q
        has_collision, self.predict_state, jacobian, self.score = self.table.apply_collision(self.state)
        if has_collision:
            self.F = jacobian
        else:
            self.F = self.system.F
            self.predict_state = self.system.f(self.state, self.u)

    def update(self, measure):
        # measurement residual
        H = np.zeros((3, 6))
        H[0][0] = H[1][1] = H[2][4] = 1
        self.y = measure - np.array([self.state[0], self.state[1], self.state[4]])
        if abs(self.y[2]) > pi:
            self.y[2] = self.y[2] - np.sign(measure[2])*2*pi
        self.S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ lg.inv(self.S)
        self.state = self.predict_state + K @ self.y
        self.P = (np.eye(6) - K @ H) @ self.P


system = air_hockey_baseline.SystemModel(tableDamping=0.001, tableFriction=0.001, tableLength=1.948, tableWidth=1.038,
                                         goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
                                         tableRes=0.7424, malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                           restitution=0.7424, rimFriction=0.1418, dt=1 / 120)
R = np.zeros((3, 3))
R[0][0] = 2.5e-7
R[1][1] = 2.5e-7
R[2][2] = 9.1e-3
Q = np.zeros((6, 6))
Q[0][0] = Q[1][1] = 2e-10
Q[2][2] = Q[3][3] = 3e-7
Q[4][4] = 1.0e-2
Q[5][5] = 1.0e-1
P = np.eye(6) * 0.01
pre_data = np.load("example_data.npy")
data = []
for i in range(1, len(pre_data)):
    if abs(pre_data[i][0] - pre_data[i - 1][0]) < 0.005 and abs(pre_data[i][1] - pre_data[i - 1][1]) < 0.005:
        continue
    data.append(pre_data[i])
data = np.array(data)
for i_data in data:
    i_data[0] += table.m_length / 2
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
state = np.array([data[1][0], data[1][1], state_dx, state_dy, data[1][2], state_dtheta])
puck_EKF = air_hockey_EKF(state=state, u=1 / 120, system=system, table=table, Q=Q, R=R, P=P)
resx = [state[0]]
resy = [state[1]]
time_EKF = [1/120]
j = 1
length = len(data)-1
i = 0
# for i in range(1, length):
while j < length:
    i += 1
    time_EKF.append((i + 1) / 120)
    if not puck_EKF.score:
        puck_EKF.predict()
        resx.append(puck_EKF.predict_state[0])
        resy.append(puck_EKF.predict_state[1])
        # check whether data is recorded at right time
        if (i-0.2) / 120 < abs(data[j+1][-1]-data[1][-1]) < (i+0.2) / 120:
            if abs(data[j+1][2] - data[j][2]) > pi:
                tmp = data[j+1][2]
                data[j+1][2] += -np.sign(data[j+1][2])*pi + data[j][2]
                puck_EKF.update(np.array(data[j + 1][0:3]))
                data[j + 1][2] = tmp
            else:
                puck_EKF.update(np.array(data[j + 1][0:3]))
            j += 1
        else:
            if abs(data[j+1][-1]-data[1][-1]) < (i-0.2) / 120:
                j += 1
            puck_EKF.state = puck_EKF.predict_state
    else:
        if abs(data[j - 1][2] - data[j][2]) > pi:
            rotation_velocity = (data[j][2] - np.sign(data[j][2])*pi) / (data[j][-1] - data[j - 1][-1])
        else:
            rotation_velocity = (data[j - 1][2] - data[j][2]) / (data[j - 1][3] - data[j][3])
        puck_EKF.state = np.array(
            [data[j][0], data[j][1], (data[j - 1][0] - data[j][0]) / (data[j - 1][3] - data[j][3]),
             (data[j - 1][1] - data[j][1]) / (data[j - 1][3] - data[j][3]), data[j][2], rotation_velocity])
        puck_EKF.predict()
        resx.append(puck_EKF.predict_state[0])
        resy.append(puck_EKF.predict_state[1])
        j += 1
table_plot(table)
plt.plot(resx[0], resy[0], marker='d', color='r')
plt.scatter(data[1:, 0], data[1:, 1], color='g', label='raw data', s=5)
plt.scatter(resx, resy, color='b', label='EKF', s=5)
plt.legend()
plt.show()
plt.subplot(2, 3, 1)
plt.scatter(time_EKF, resx, color='b', label='EKF x position', s=5)
plt.title('only EKF x position')
plt.legend()
plt.subplot(2, 3, 2)
plt.scatter(data[1:, -1]-data[0][-1], data[1:, 0], color='g', label='raw data x position', s=5)
plt.title('only raw data x position')
plt.legend()
plt.subplot(2, 3, 3)
plt.scatter(time_EKF, resx, color='b', label='EKF x position', s=5)
plt.scatter(data[1:, -1]-data[0][-1], data[1:, 0], color='g', label='raw data x position', s=5)
plt.title('EKF vs raw data x position')
plt.legend()
plt.subplot(2, 3, 4)
plt.scatter(time_EKF, resy, color='b', label='EKF y position', s=5)
plt.title('only EKF y position')
plt.legend()
plt.subplot(2, 3, 5)
plt.scatter(data[1:, -1]-data[0][-1], data[1:, 1], color='g', label='raw data y position', s=5)
plt.title('only raw data y position')
plt.legend()
plt.subplot(2, 3, 6)
plt.scatter(time_EKF, resy, color='b', label='EKF y position', s=5)
plt.scatter(data[1:, -1]-data[0][-1], data[1:, 1], color='g', label='raw data y position', s=5)
plt.title('EKF vs raw data y position')
plt.legend()
plt.show()
