from EKF_Wrapper import air_hockey_EKF
import pandas as pd
from hebo.design_space.design_space import DesignSpace
import air_hockey_baseline
import numpy as np
import matplotlib.pyplot as plt
from hebo.optimizers.bo import BO
from hebo.optimizers.hebo import HEBO
from math import pi

params = [{'name': 'tableFriction', 'type': 'num', 'lb': 0, 'ub': 0.5},
          {'name': 'tableDamping', 'type': 'num', 'lb': 0, 'ub': 0.5},
          {'name': 'tableRestitution', 'type': 'num', 'lb': 0, 'ub': 0.9}]
space = DesignSpace().parse(params)
bo = BO(space)
hebo = HEBO(space, rand_sample=5)


def obj(x: pd.DataFrame) -> np.ndarray:
    x = x[['tableFriction', 'tableDamping', 'tableRestitution']].values
    num_x = x.shape[0]
    ret = np.zeros((num_x, 1))
    for k in range(num_x):
        ret[k, 0] = expectation(x[k])
    return ret


def expectation(Nparams):
    # system initialization
    system = air_hockey_baseline.SystemModel(tableDamping=Nparams[1], tableFriction=Nparams[0],
                                             tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                             puckRadius=0.03165, malletRadius=0.04815, tableRes=Nparams[2],
                                             malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
    table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                               restitution=Nparams[2], rimFriction=0.1418, dt=1 / 120)
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
    raw_data = np.load("example_data2.npy")
    pre_data = []
    for j in range(1, len(raw_data)):
        if abs(raw_data[j][0] - raw_data[j - 1][0]) < 0.005 and abs(raw_data[j][1] - raw_data[j - 1][1]) < 0.005:
            continue
        pre_data.append(raw_data[j])
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
    state = np.array([pre_data[1][0], pre_data[1][1], state_dx, state_dy, pre_data[1][2], state_dtheta])
    data = np.array(pre_data[1:])
    u = 1 / 120
    # EKF start
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
                    data[j][2] += -np.sign(data[j][2]) * pi + data[j - 1][2]
                    puck_EKF.update(np.array(data[j][0:3]))
                    data[j][2] = tmp
                else:
                    puck_EKF.update(np.array(data[j][0:3]))
                j += 1
                sign, logdet = np.linalg.slogdet(puck_EKF.S)
                evaluation += (sign*np.exp(logdet) + puck_EKF.y.T @ np.linalg.inv(puck_EKF.S) @ puck_EKF.y)
                num_evaluation += 1

            else:
                if abs(data[j][-1] - data[0][-1]) <= (i - 0.2) / 120:
                    j += 1
                puck_EKF.state = puck_EKF.predict_state
        else:
            if abs(data[j - 1][2] - data[j - 2][2]) > pi:
                rotation_velocity = (data[j - 1][2] - np.sign(data[j - 1][2]) * pi) / (
                        data[j - 1][-1] - data[j - 2][-1])
            else:
                rotation_velocity = (data[j - 1][2] - data[j - 2][2]) / (data[j - 1][3] - data[j - 2][3])
            puck_EKF.state = np.array(
                [data[j - 1][0], data[j - 1][1], (data[j - 1][0] - data[j - 2][0]) / (data[j - 1][3] - data[j - 2][3]),
                 (data[j - 1][1] - data[j - 2][1]) / (data[j - 1][3] - data[j - 2][3]), data[j - 1][2],
                 rotation_velocity])
            puck_EKF.predict()
            j += 1
    return evaluation/num_evaluation


for i in range(30):
    rec_x = hebo.suggest(n_suggestions=5)

    hebo.observe(rec_x, obj(rec_x))
    min_idx = hebo.y.argmin()
    print(str(i+1) + ' y is ' + str(hebo.y.min()) + ' tableFriction ' + str(hebo.X.iloc[min_idx]['tableFriction']) +' tableDamping '+ str(
        hebo.X.iloc[min_idx]['tableDamping']) + ' tableRestitution ' + str(hebo.X.iloc[min_idx]['tableRestitution']))
    plt.scatter(i+1, hebo.y.min(), color='b')
plt.show()
