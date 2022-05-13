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
        self.F = self.system.F
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
'''
z = np.array([0.51, 0.001, 1])
one step test
state = np.array([1.85, 0.3, 5.0, 0, 0, 0])
y = []
puck = air_hockey_EKF(state=state, u=1 / 120, system=system, table=table, Q=Q, R=R, P=P)
puck.predict()
puck.update(np.array([1.89, 0.3, 0]))
puck.predict()
print()
'''
data = np.load("example_data.npy")
orgx = []
orgy = []
for i in data:
    i[0] += table.m_length / 2
    orgx.append(i[0])
    orgy.append(i[1])
state = np.array([data[0][0], data[0][1], (data[1][0] - data[0][0]) / (data[1][3] - data[0][3]),
                  (data[1][1] - data[0][1]) / (data[1][3] - data[0][3]), data[0][3],
                  (data[1][2] - data[0][2]) / (data[1][3] - data[0][3])])
puck_EKF = air_hockey_EKF(state=state, u=1 / 120, system=system, table=table, Q=Q, R=R, P=P)
resx = [state[0]]
resy = [state[1]]
start_t = data[0][-1]
for i in range(len(data) - 1):
    puck_EKF.predict()
    resx.append(puck_EKF.predict_state[0])
    resy.append(puck_EKF.predict_state[1])
    if i > 0 and abs(data[i][-1] - data[i - 1][-1]) > 0.8 / 120:
        puck_EKF.update(np.array(data[i + 1][0:3]))
    else:
        puck_EKF.state = puck_EKF.predict_state
table_plot(table)
plt.plot(resx[0], resy[0], marker='d', color='r')
plt.plot(orgx, orgy, color='g', label='raw data')
plt.plot(resx, resy, color='b', label='EKF')
plt.legend()
plt.show()
plt.subplot(1, 3, 1)
plt.plot(resx, resy, color='b', label='EKF')
plt.title('only EKF')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(orgx, orgy, color='g', label='raw data')
plt.title('only raw data')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(orgx, orgy, color='g', label='raw data')
plt.plot(resx, resy, color='b', label='EKF')
plt.title('EKF vs raw data')
plt.legend()
plt.show()
