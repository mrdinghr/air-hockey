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
data = np.load("example_data1.npy")
orgx = []
orgy = []
for i in data:
    i[0] += table.m_length / 2
    orgx.append(i[0])
    orgy.append(i[1])
state = np.array([data[0][0], data[0][1], (data[1][0] - data[0][0]) / (data[1][3] - data[0][3]),
                  (data[1][1] - data[0][1]) / (data[1][3] - data[0][3]), data[0][3],
                  (data[1][2] - data[0][2]) / (data[1][3] - data[0][3])])
u = 1 / 120
puck_EKF = air_hockey_EKF(state=state, u=u, system=system, table=table, Q=Q, R=R, P=P)
EKF__res_state = []
EKF_res_P = []
EKF_res_dynamic = []
start_t = data[0][-1]
for i in range(len(data) - 1):
    if not puck_EKF.score:
        puck_EKF.predict()
        EKF__res_state.append(puck_EKF.predict_state)
        EKF_res_P.append(puck_EKF.P)
        EKF_res_dynamic.append(puck_EKF.F)
        if i > 0 and abs(data[i][-1] - data[i - 1][-1]) > 0.8 / 120:
            puck_EKF.update(np.array(data[i + 1][0:3]))
        else:
            puck_EKF.state = puck_EKF.predict_state
    else:
        puck_EKF.state = np.array(
            [data[i][0], data[i][1], (data[i - 1][0] - data[i][0]) / (data[i - 1][3] - data[i][3]),
             (data[i - 1][1] - data[i][1]) / (data[i - 1][3] - data[i][3]), data[i][3],
             (data[i - 1][2] - data[i][2]) / (data[i - 1][3] - data[i][3])])
        puck_EKF.predict()
'''
Kalman Smoothing
F： dynamic jacobian as in EKF

xp_n+1=x_n*F   x_n EKF predicted state, p_n
p_p_n+1=F*p_n+Q 
C_n=p_n*F.T*inv(P_p_n+1)
'''
smooth_res_state = [EKF__res_state[-1]]
xs = EKF__res_state[-1]
time = np.shape(EKF__res_state)[0]
for i in range(time - 2):
    xp = EKF_res_dynamic[-i - 2] @ EKF__res_state[-i - 2]
    pp = EKF_res_dynamic[-i - 2] @ EKF_res_P[-i - 2] @ EKF_res_dynamic[-i - 2].T
    c = EKF_res_P[-i - 2]
    xs = EKF__res_state[-i - 2] + c @ (xs - xp)
    smooth_res_state.append(xs)
resx = []
resy = []
for t in EKF__res_state:
    resx.append(t[0])
    resy.append(t[1])
plt.plot(resx, resy)
plt.show()
smooth_res_x = []
smooth_res_y = []
for m in smooth_res_state:
    smooth_res_x.insert(0, m[0])
    smooth_res_y.insert(0, m[1])
plt.plot(smooth_res_x, smooth_res_x)
plt.show()
