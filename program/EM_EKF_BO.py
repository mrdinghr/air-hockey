from EKF_Wrapper import air_hockey_EKF
import pandas as pd
from hebo.design_space.design_space import DesignSpace
import torch
import air_hockey_baseline
import numpy as np

params = [{'name': 'tableFriction', 'type': 'num', 'lb': 0, 'ub': 0.9},
          {'name': 'tableDamping', 'type': 'num', 'lb': 0, 'ub': 0.9},
          {'name': 'tableRestitution', 'type': 'num', 'lb': 0, 'ub': 0.9}]
space = DesignSpace().parse(params)
samp = space.sample(27)
samp_params = space.transform(samp)
Nparams = samp_params[0]


def expectation(Nparams):
    for i in range(len(Nparams)):
        system = air_hockey_baseline.SystemModel(tableDamping=Nparams[i][1], tableFriction=Nparams[i][0],
                                                 tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                 puckRadius=0.03165, malletRadius=0.04815, tableRes=Nparams[i][2],
                                                 malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
        table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                                   restitution=Nparams[i][2], rimFriction=0.1418, dt=1 / 120)
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
        for j in range(1, len(raw_data)):
            if abs(raw_data[j][0] - raw_data[j - 1][0]) < 0.005 and abs(raw_data[j][1] - raw_data[j - 1][1]) < 0.005:
                continue
            pre_data.append(raw_data[j])
        for m in pre_data:
            m[0] += table.m_length / 2
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
        data = pre_data[1:]
        u = 1 / 120
        puck_EKF = air_hockey_EKF(state=state, u=u, system=system, table=table, Q=Q, R=R, P=P)
