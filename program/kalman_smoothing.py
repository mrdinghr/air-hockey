import numpy.linalg as lg
import numpy as np
import air_hockey_baseline
import matplotlib.pyplot as plt
from air_hockey_plot import table_plot
from math import pi
from EKF_Wrapper import air_hockey_EKF


def kalman_smooth(raw_data, system, table):
    R = np.zeros((3, 3))
    R[0][0] = 2.5e-7
    R[1][1] = 2.5e-7
    R[2][2] = 9.1e-3
    Q = np.zeros((6, 6))
    Q[0][0] = Q[1][1] = 2e-10
    Q[2][2] = Q[3][3] = 3e-7
    Q[4][4] = 1.0e-2
    Q[5][5] = 5
    P = np.eye(6) * 0.01
    pre_data = []
    for i in range(1, len(raw_data)):
        if abs(raw_data[i][0] - raw_data[i - 1][0]) < 0.005 and abs(raw_data[i][1] - raw_data[i - 1][1]) < 0.005:
            continue
        pre_data.append(raw_data[i])
    for i in pre_data:
        i[0] += table.m_length / 2
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
    puck_EKF = air_hockey_EKF(state=state, u=u, system=system, table=table, Q=Q, R=R, P=P)
    EKF_res_state = []
    EKF_res_P = []
    EKF_res_dynamic = []
    EKF_res_score = []
    EKF_res_collision = []
    EKF_res_update = []
    i = 0
    j = 1
    length = len(data)
    time_EKF = []
    while j < length:
        i += 1
        time_EKF.append(i / 120)
        if not puck_EKF.score:
            puck_EKF.predict()
            EKF_res_score.append(False)
            EKF_res_state.append(puck_EKF.predict_state)
            EKF_res_P.append(puck_EKF.P)
            EKF_res_dynamic.append(puck_EKF.F)
            EKF_res_collision.append(puck_EKF.has_collision)
            if (i - 0.2) / 120 < abs(data[j][-1] - data[0][-1]) < (i + 0.2) / 120:
                EKF_res_update.append(True)
                if abs(data[j - 1][2] - data[j][2]) > pi:
                    tmp = data[j][2]
                    data[j][2] += -np.sign(data[j][2]) * pi + data[j - 1][2]
                    puck_EKF.update(np.array(data[j][0:3]))
                    data[j][2] = tmp
                else:
                    puck_EKF.update(np.array(data[j][0:3]))
                j += 1
            else:
                EKF_res_update.append(False)
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
            EKF_res_state.append(puck_EKF.predict_state)
            EKF_res_P.append(puck_EKF.P)
            EKF_res_dynamic.append(puck_EKF.F)
            EKF_res_score.append(True)
            EKF_res_update.append(False)
            EKF_res_collision.append(puck_EKF.has_collision)
            j += 1
    EKF_res_state = np.array(EKF_res_state)
    '''
    Kalman Smoothing
    F： dynamic jacobian as in EKF
    
    xp_n+1=x_n*F   x_n EKF predicted state, p_n
    p_p_n+1=F*p_n+Q 
    C_n=p_n*F.T*inv(P_p_n+1)
    '''
    smooth_res_state = [EKF_res_state[-1]]
    xs = EKF_res_state[-1]
    ps = EKF_res_P[-1]
    time = np.shape(EKF_res_state)[0]
    xp = np.zeros(6)
    evaluation = 0
    num_evaluation = 0
    i = 0
    for j in range(time - 1):
        if EKF_res_update[-1 - j]:
            i += 1
            innovation = data[-i, 0:3] - np.array([xs[0], xs[1], xs[4]])
            innovation_covariance = puck_EKF.H@ps@puck_EKF.H.T + R
            sign, logdet = np.linalg.slogdet(innovation_covariance)
            num_evaluation += 1
            evaluation += (sign*np.exp(logdet) + innovation.T @ np.linalg.inv(innovation_covariance) @ innovation)
        if not EKF_res_score[-2 - j]:
            xp = EKF_res_dynamic[-j - 1] @ EKF_res_state[-j - 2]
            if not EKF_res_collision[-j - 1]:
                if np.sqrt(EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
                           EKF_res_state[-j - 2][3]) > 1e-6:
                    xp[2:4] = EKF_res_state[-j - 2][2:4] - u * (
                            system.tableDamping * EKF_res_state[-j - 2][2:4] + system.tableFriction * EKF_res_state[
                                                                                                          -j - 2][
                                                                                                      2:4] / np.sqrt(
                        EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
                        EKF_res_state[-j - 2][3]))
                else:
                    xp[2:4] = EKF_res_state[-j - 2][2:4] - u * system.tableDamping * EKF_res_state[-j - 2][2:4]
            pp = EKF_res_dynamic[-j - 1] @ EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T + Q
            c = EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T @ lg.inv(pp)
            if abs(xs[4] - xp[4]) > pi:
                xp[4] = xp[4] - np.sign(xp[4]) * 2 * pi
            if xs[5] * xp[5] < 0:
                xs[5] = -xs[5]
                # smooth_res_state[-1][5] *= -1
            xs = EKF_res_state[-j - 2] + c @ (xs - xp)
            ps = EKF_res_P[-j - 2] + c @ (ps - pp) @ c.T
            smooth_res_state.append(xs)
        else:
            xs = EKF_res_state[-j - 2]
            ps = EKF_res_P[-j - 2]
            xp = EKF_res_dynamic[-j - 1] @ EKF_res_state[-j - 2]
            smooth_res_state.append(xs)
    return evaluation/num_evaluation
    # follow code is used to test plot result of kalman smooth. so it s nothing to do with EM process.
    '''
    smooth_res_state = np.array(smooth_res_state)
    table_plot(table)
    plt.plot(EKF_res_state[0][0], EKF_res_state[0][1], marker='d', color='r')
    plt.scatter(data[:, 0], data[:, 1], color='g', label='raw data', s=5)
    plt.scatter(EKF_res_state[:, 0], EKF_res_state[:, 1], color='b', label='EKF', s=5)
    plt.scatter(smooth_res_state[:, 0], smooth_res_state[:, 1], color='r', label='smooth', s=5)
    plt.legend()
    # plt.show()
    # calculate raw data velocity
    data_x_velocity = []
    data_y_velocity = []
    data_theta_velocity = []
    for i in range(1, len(pre_data)):
        data_x_velocity.append((pre_data[i][0] - pre_data[i - 1][0]) / (pre_data[i][-1] - pre_data[i - 1][-1]))
        data_y_velocity.append((pre_data[i][1] - pre_data[i - 1][1]) / (pre_data[i][-1] - pre_data[i - 1][-1]))
        if abs(pre_data[i][2] - pre_data[i - 1][2]) > pi:
            data_theta_velocity.append(
                (pre_data[i][2] - np.sign(pre_data[i][2]) * pi) / (pre_data[i][-1] - pre_data[i - 1][-1]))
        else:
            data_theta_velocity.append((pre_data[i][2] - pre_data[i - 1][2]) / (pre_data[i][-1] - pre_data[i - 1][-1]))
    # plot x position
    plt.subplot(3, 4, 1)
    plt.scatter(time_EKF, EKF_res_state[:, 0], color='b', label='EKF x position', s=5)
    plt.title('only EKF x position')
    plt.legend()
    plt.subplot(3, 4, 2)
    plt.scatter(data[:, -1] - data[0][-1], data[:, 0], color='g', label='raw data x position', s=5)
    plt.title('only raw data x position')
    plt.legend()
    plt.subplot(3, 4, 3)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 0], color='r', label='smooth x position', s=5)
    plt.title('smooth x position')
    plt.legend()
    plt.subplot(3, 4, 4)
    plt.scatter(time_EKF, EKF_res_state[:, 0], color='b', label='EKF x position', s=5)
    plt.scatter(data[:, -1] - data[0][-1], data[:, 0], color='g', label='raw data x position', s=5)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 0], color='r', label='smooth x position', s=5)
    plt.legend()
    # another line to plot y position
    plt.subplot(3, 4, 5)
    plt.scatter(time_EKF, EKF_res_state[:, 1], color='b', label='EKF y position', s=5)
    plt.title('only EKF y position')
    plt.legend()
    plt.subplot(3, 4, 6)
    plt.scatter(data[:, -1] - data[0][-1], data[:, 1], color='g', label='raw data y position', s=5)
    plt.title('only raw data y position')
    plt.legend()
    plt.subplot(3, 4, 7)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 1], color='r', label='smooth y position', s=5)
    plt.title('smooth y position')
    plt.legend()
    plt.subplot(3, 4, 8)
    plt.scatter(time_EKF, EKF_res_state[:, 1], color='b', label='EKF y position', s=5)
    plt.scatter(data[:, -1] - data[0][-1], data[:, 1], color='g', label='raw data y position', s=5)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 1], color='r', label='smooth y position', s=5)
    plt.legend()
    # plot theta
    plt.subplot(3, 4, 10)
    plt.scatter(data[:, -1] - data[0][-1], data[:, 2], color='g', label='raw data theta', s=5)
    plt.title('only raw data  theta')
    plt.legend()
    plt.subplot(3, 4, 11)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 4], color='r', label='smooth theta', s=5)
    plt.title('smooth theta')
    plt.legend()
    plt.subplot(3, 4, 9)
    plt.scatter(time_EKF, EKF_res_state[:, 4], color='b', label='EKF theta', s=5)
    plt.title('only EKF theta')
    plt.legend()
    plt.subplot(3, 4, 12)
    plt.scatter(time_EKF, EKF_res_state[:, 4], color='b', label='EKF theta', s=5)
    plt.scatter(data[:, -1] - data[0][-1], data[:, 2], color='g', label='raw data theta', s=5)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 4], color='r', label='smooth theta', s=5)
    plt.legend()
    plt.figure()
    # plt.show()
    # x velocity
    plt.subplot(3, 4, 1)
    plt.scatter(time_EKF, EKF_res_state[:, 2], color='b', label='EKF x velocity', s=5)
    plt.title('EKF x velocity')
    plt.legend()
    plt.subplot(3, 4, 2)
    plt.scatter(data[:, -1] - data[0][-1], data_x_velocity, color='g', label='raw data x velocity', s=5)
    plt.title('raw data x velocity')
    plt.legend()
    plt.subplot(3, 4, 3)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 2], color='r', label='smooth x velocity', s=5)
    plt.title('smooth x velocity')
    plt.legend()
    plt.subplot(3, 4, 4)
    plt.scatter(time_EKF, EKF_res_state[:, 2], color='b', label='EKF x velocity', s=5)
    plt.scatter(data[:, -1] - data[0][-1], data_x_velocity, color='g', label='raw data x velocity', s=5)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 2], color='r', label='smooth x velocity', s=5)
    plt.legend()
    # y velocity
    plt.subplot(3, 4, 5)
    plt.scatter(time_EKF, EKF_res_state[:, 3], color='b', label='EKF y velocity', s=5)
    plt.title('EKF y velocity')
    plt.legend()
    plt.subplot(3, 4, 6)
    plt.scatter(data[:, -1] - data[0][-1], data_y_velocity, color='g', label='raw data y velocity', s=5)
    plt.title('raw data y velocity')
    plt.legend()
    plt.subplot(3, 4, 7)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 3], color='r', label='smooth y velocity', s=5)
    plt.title('smooth y velocity')
    plt.legend()
    plt.subplot(3, 4, 8)
    plt.scatter(time_EKF, EKF_res_state[:, 3], color='b', label='EKF x velocity', s=5)
    plt.scatter(data[:, -1] - data[0][-1], data_y_velocity, color='g', label='raw data x velocity', s=5)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 3], color='r', label='smooth x velocity', s=5)
    plt.legend()
    # rotation velocity
    plt.subplot(3, 4, 9)
    plt.scatter(time_EKF, EKF_res_state[:, 5], color='b', label='EKF rotation velocity', s=5)
    plt.title('EKF rotation velocity')
    plt.legend()
    plt.subplot(3, 4, 10)
    plt.scatter(data[:, -1] - data[0][-1], data_theta_velocity, color='g', label='raw data rotation velocity', s=5)
    plt.title('raw data rotation velocity')
    plt.legend()
    plt.subplot(3, 4, 11)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 5], color='r', label='smooth rotation velocity', s=5)
    plt.title('smooth rotation velocity')
    plt.legend()
    plt.subplot(3, 4, 12)
    plt.scatter(time_EKF, EKF_res_state[:, 5], color='b', label='EKF x velocity', s=5)
    plt.scatter(data[:, -1] - data[0][-1], data_theta_velocity, color='g', label='raw data x velocity', s=5)
    plt.scatter(time_EKF[:], smooth_res_state[-1::-1, 5], color='r', label='smooth x velocity', s=5)
    plt.legend()
    plt.show()
'''


'''
raw_data = np.load("example_data.npy")
system = air_hockey_baseline.SystemModel(tableDamping=0.001, tableFriction=0.001, tableLength=1.948, tableWidth=1.038,
                                         goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
                                         tableRes=0.7424, malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                           restitution=0.7424, rimFriction=0.1418, dt=1 / 120)
kalman_smooth(raw_data, system, table)
'''
