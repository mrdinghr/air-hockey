import numpy.linalg as lg
import numpy as np
import air_hockey_baseline
import matplotlib.pyplot as plt
from air_hockey_plot import table_plot


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
        self.has_collision = False

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
        H = np.zeros((3, 6))
        H[0][0] = H[1][1] = H[2][4] = 1
        y = measure - np.array([self.state[0], self.state[1], self.state[4]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ lg.inv(S)
        self.state = self.predict_state + K @ y
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
raw_data = np.load("example_data.npy")
pre_data = []
for i in range(1, len(raw_data)):
    if abs(raw_data[i][0] - raw_data[i - 1][0]) < 0.005 and abs(raw_data[i][1] - raw_data[i - 1][1]) < 0.005:
        continue
    pre_data.append(raw_data[i])
orgx = []
orgy = []
for i in pre_data:
    i[0] += table.m_length / 2
    orgx.append(i[0])
    orgy.append(i[1])
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
state = np.array([pre_data[1][0], pre_data[1][1], state_dx, state_dy, pre_data[1][2], state_dtheta])
# state = np.array([data[0][0], data[0][1], (data[1][0] - data[0][0]) / (data[1][3] - data[0][3]),
#                   (data[1][1] - data[0][1]) / (data[1][3] - data[0][3]), data[0][2],
#                   (data[1][2] - data[0][2]) / (data[1][3] - data[0][3])])
data = pre_data[1:]

u = 1 / 120
puck_EKF = air_hockey_EKF(state=state, u=u, system=system, table=table, Q=Q, R=R, P=P)
EKF_res_state = []
EKF_res_P = []
EKF_res_dynamic = []
EKF_res_score = []
EKF_res_collision = []
for i in range(1, len(data)):
    if not puck_EKF.score:
        puck_EKF.predict()
        EKF_res_score.append(False)
        EKF_res_state.append(puck_EKF.predict_state)
        EKF_res_P.append(puck_EKF.P)
        EKF_res_dynamic.append(puck_EKF.F)
        EKF_res_collision.append(puck_EKF.has_collision)
        if 1.2 / 120 > abs(data[i][-1] - data[i - 1][-1]) > 0.8 / 120:
            puck_EKF.update(np.array(data[i+1][0:3]))

        else:
            puck_EKF.state = puck_EKF.predict_state
        #     EKF_res_state.append(puck_EKF.predict_state)
        #     EKF_res_P.append(puck_EKF.P)
        #     EKF_res_dynamic.append(puck_EKF.F)
        #     EKF_res_collision.append(puck_EKF.has_collision)
    else:
        puck_EKF.state = np.array(
            [data[i][0], data[i][1], (data[i - 1][0] - data[i][0]) / (data[i - 1][3] - data[i][3]),
             (data[i - 1][1] - data[i][1]) / (data[i - 1][3] - data[i][3]), data[i][2],
             (data[i - 1][2] - data[i][2]) / (data[i - 1][3] - data[i][3])])
        puck_EKF.predict()
        EKF_res_state.append(puck_EKF.predict_state)
        EKF_res_P.append(puck_EKF.P)
        EKF_res_dynamic.append(puck_EKF.F)
        EKF_res_score.append(True)
        EKF_res_collision.append(puck_EKF.has_collision)
'''
Kalman Smoothing
F： dynamic jacobian as in EKF

xp_n+1=x_n*F   x_n EKF predicted state, p_n
p_p_n+1=F*p_n+Q 
C_n=p_n*F.T*inv(P_p_n+1)
'''
smooth_res_state = [EKF_res_state[-1]]
xs = EKF_res_state[-1]
time = np.shape(EKF_res_state)[0]
xp = np.zeros(6)
for j in range(time - 2):
    if not EKF_res_score[-2 - j]:
        xp = EKF_res_dynamic[-j - 1] @ EKF_res_state[-j - 2]
        if not EKF_res_collision[-j - 1]:
            if np.sqrt(EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
                       EKF_res_state[-j - 2][3]) > 1e-6:
                xp[2:4] = EKF_res_state[-j - 2][2:4] - u * (
                        system.tableDamping * EKF_res_state[-j - 2][2:4] + system.tableFriction * EKF_res_state[-j - 2][
                                                                                                  2:4] / np.sqrt(
                    EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
                    EKF_res_state[-j - 2][3]))
            else:
                xp[2:4] = EKF_res_state[-j - 2][2:4] - u * system.tableDamping * EKF_res_state[-j - 2][2:4]
        pp = EKF_res_dynamic[-j - 1] @ EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T + Q
        c = EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T @ lg.inv(pp)
        xs = EKF_res_state[-j - 2] + c @ (xs - xp)
        smooth_res_state.append(xs)
    else:
        xs = EKF_res_state[-j - 2]
        xp = EKF_res_dynamic[-j - 1] @ EKF_res_state[-j - 2]
        smooth_res_state.append(xs)
resx = []
resy = []
for t in EKF_res_state:
    resx.append(t[0])
    resy.append(t[1])
smooth_res_x = []
smooth_res_y = []
for m in smooth_res_state:
    smooth_res_x.insert(0, m[0])
    smooth_res_y.insert(0, m[1])
plt.subplot(1, 3, 1)
plt.plot(orgx[1], orgy[1], marker='d', color='r')
plt.scatter(orgx[1:], orgy[1:], color='r', label='Raw Data')
plt.title('raw data')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(orgx[1], orgy[1], marker='d', color='r')
plt.scatter(smooth_res_x, smooth_res_y, color='b', label='Kalman Smooth')
plt.title('kalman smooth')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(orgx[1], orgy[1], marker='d', color='r')
plt.scatter(resx, resy, color='g', label='EKF')
plt.title('EKF')
plt.legend()
plt.show()
plt.plot(orgx[1], orgy[1], marker='d', color='r')
plt.scatter(resx, resy, color='g', label='EKF')
plt.scatter(orgx[4:], orgy[4:], color='r', label='Raw Data')
plt.scatter(smooth_res_x, smooth_res_y, color='b', label='Kalman Smooth')
plt.legend()
plt.show()
# next plot x y with time
plt.subplot(1, 2, 1)
plt.plot(orgx[4:], label='raw data')
plt.plot(resx, label='EKF')
plt.plot(smooth_res_x, label='kalman smooth')
plt.legend()
plt.title('x')
plt.subplot(1, 2, 2)
plt.plot(orgy[4:], label='raw data')
plt.plot(resy, label='EKF')
plt.plot(smooth_res_y, label='kalman smooth')
plt.title('y')
plt.legend()
plt.show()
