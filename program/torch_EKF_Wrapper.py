from math import pi
import torch
import torch_air_hockey_baseline_no_detach
from matplotlib import pyplot as plt
import numpy as np
from test_params import plot_with_state_list
from test_params import EKF_plot_with_state_list

torch.set_printoptions(threshold=torch.inf)


class AirHockeyEKF:
    def __init__(self, u, system, Q, R, P, device):
        self.state = None
        self.system = system
        self.Q = Q
        self.Q_score = torch.zeros((6, 6), device=device)
        self.Q_score[2][2] = 1
        self.Q_score[3][3] = 1
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
        self.score_time = 0
        self.device = device

    def initialize(self, state):
        self.state = state
        self.P = torch.eye(6, device=self.device).float() * 0.01

    def reset_angle(self, state):
        while state[4] > pi:
            state[4] = state[4] - 2 * pi
        while state[4] < -pi:
            state[4] = state[4] + 2 * pi

    # params: table friction, table damping, table restitution, rim friction
    def predict(self, beta=0, res=None):
        self.has_collision, self.predict_state, jacobian, self.score = self.system.apply_collision(self.state,
                                                                                                   beta=beta)
        if self.has_collision:
            self.F = jacobian.clone()
        else:
            self.F = self.system.F.clone()
            self.predict_state = self.system.f(self.state, self.u)
        if res is not None:
            self.F = self.F + torch.autograd.functional.jacobian(res.cal_res, self.state)
            self.predict_state = self.predict_state + res.cal_res(self.state)
        self.reset_angle(self.predict_state)
        self.P = self.F @ self.P @ self.F.T + self.Q
        # if self.score or self.score_time != 0:
        #     self.score_time += 1
        #     # self.predict_state = self.state + 0 * self.system.tableDamping
        #     self.P = self.P + self.Q_score
        #     if self.score_time == 5:
        #         self.score_time = 0

    def update(self, measure, update=True):
        # measurement residual
        self.y = measure - self.predict_state[[0, 1, 4]]
        if self.y[2] >= pi:
            self.y[2] = self.y[2] - pi * 2
        elif self.y[2] <= -pi:
            self.y[2] = self.y[2] + pi * 2
        self.S = self.H @ self.P @ self.H.T + self.R
        # self.S.requires_grad_(True)
        K = self.P @ self.H.T @ torch.linalg.inv(self.S)
        if update:
            self.state = self.predict_state + K @ self.y
            self.P = (torch.eye(6, device=self.device) - K @ self.H) @ self.P
        else:
            self.state = self.predict_state
            self.P = self.P

    def refresh(self, P, Q, R):
        self.P = P
        self.Q = Q
        self.R = R
        self.F = None
        self.predict_state = None
        self.y = None
        self.S = None
        self.u = 1 / 120
        self.score = False
        self.has_collision = False
        self.H = torch.zeros((3, 6), device=self.device)
        self.H[0][0] = self.H[1][1] = self.H[2][4] = 1
        self.state = None

    def smooth(self, init_state, trajectory, plot=False, writer=None, epoch=0, trajectory_index=None,
               cal=None, beta=0, res=None):
        state_list, variance_list, jacobian_list, collision_list, update_list = self.forward_pass(init_state,
                                                                                                  trajectory,
                                                                                                  beta=beta,
                                                                                                  cal=cal,
                                                                                                  res=res)
        smoothed_state_list, smoothed_variance_list = self.backward_pass(state_list, variance_list, jacobian_list,
                                                                         update_list, beta=beta,
                                                                         cal=cal, res=res)
        if plot:
            time_list = [i / 120 for i in range(len(state_list))]
            plot_with_state_list(state_list, smoothed_state_list, trajectory, time_list, writer=writer, epoch=epoch,
                                 trajectory_index=trajectory_index)
        smoothed_state_list = smoothed_state_list[::-1]
        smoothed_variance_list = smoothed_variance_list[::-1]
        collision_list = collision_list[::-1]
        return smoothed_state_list, smoothed_variance_list, collision_list

    def forward_pass(self, init_state, trajectory, beta=0, cal=None, update=True,
                     res=None):
        self.initialize(init_state)
        EKF_res_state = [init_state]
        EKF_res_P = [self.P]
        EKF_res_dynamic = [self.system.F]
        EKF_res_collision = [False]
        EKF_res_update = [False]
        i = 0
        j = 0
        length = len(trajectory)
        while j < length - 1:
            i += 1
            if cal is not None:
                params = cal.cal_params(self.state[0:2])
                self.system.set_params(tableDampingX=params[0], tableDampingY=params[1], tableFrictionX=params[2],
                                       tableFrictionY=params[3], restitution=params[4],
                                       rimFriction=params[5])
            self.predict(beta=beta, res=res)
            EKF_res_state.append(self.predict_state)
            EKF_res_P.append(self.P)
            EKF_res_dynamic.append(self.F)
            EKF_res_collision.append(self.has_collision)
            if (i - 0.5) / 120 <= trajectory[j + 1][-1] - trajectory[0][-1] <= (i + 0.5) / 120:
                self.update(trajectory[j + 1][0:3], update=update)
                j += 1
                EKF_res_update.append(1)
            elif trajectory[j + 1][-1] - trajectory[0][-1] < (i - 0.5) / 120:
                j = j + 1
                i = i - 1
                self.state = self.predict_state
                EKF_res_update.append(2)
            else:
                self.state = self.predict_state
                EKF_res_update.append(False)
        return EKF_res_state, EKF_res_P, EKF_res_dynamic, EKF_res_collision, EKF_res_update

    def backward_pass(self, state_list, variance_list, jacobian_list, update_list, beta=0, cal=None, res=None):
        smoothed_state_list = [state_list[-1]]
        smoothed_variance_list = [self.H @ variance_list[-1] @ self.H.T + self.R]
        xs = smoothed_state_list[-1]
        ps = variance_list[-1]
        time = len(state_list)
        for j in range(time - 1):
            idx_prev = - j - 2
            if cal is not None:
                params = cal.cal_params(state_list[idx_prev][:2])
                self.system.set_params(tableDampingX=params[0], tableDampingY=params[1], tableFrictionX=params[2],
                                       tableFrictionY=params[3], restitution=params[4],
                                       rimFriction=params[5])
            has_collision, predict_state, _, _ = self.system.apply_collision(state_list[idx_prev], beta=beta)
            if not has_collision:
                xp = self.system.f(state_list[idx_prev], self.u)
            else:
                xp = predict_state
            # xp = jacobian_list[idx_cur] @ state_list[idx_prev]
            # if not collision_list[idx_cur]:
            #     if torch.linalg.norm(state_list[idx_prev][2:4]) > 1e-6:
            #         xp[2:4] = state_list[idx_prev][2:4] - self.u * (system.tableDamping * state_list[idx_prev][2:4] +
            #                                                         system.tableFriction *
            #                                                         torch.sign(state_list[idx_prev][2:4]))
            #     else:
            #         xp[2:4] = state_list[idx_prev][2:4] - self.u * system.tableDamping * state_list[idx_prev][2:4]
            if xs[4] - xp[4] > 3 / 2 * pi:
                xp4 = xp[4] + 2 * pi
            elif xs[4] - xp[4] < -3 / 2 * pi:
                xp4 = xp[4] - 2 * pi
            else:
                xp4 = xp[4]
            if res is not None:
                res_state = res.cal_res(state_list[idx_prev])
                xp_new = torch.cat(
                    [xp[0:4] + res_state[0:4], torch.atleast_1d(xp4 + res_state[4]), torch.atleast_1d(xp[5] + res_state[5])])
            else:
                xp_new = torch.cat([xp[0:4], torch.atleast_1d(xp4), torch.atleast_1d(xp[5])])
            predicted_cov = jacobian_list[idx_prev] @ variance_list[idx_prev] @ jacobian_list[idx_prev].T + self.Q
            smooth_gain = variance_list[idx_prev] @ jacobian_list[idx_prev].T @ torch.linalg.inv(predicted_cov)

            xs = state_list[idx_prev] + smooth_gain @ (xs - xp_new)
            ps = variance_list[idx_prev] + smooth_gain @ (ps - predicted_cov) @ smooth_gain.T
            if update_list[idx_prev] > 0:
                smoothed_state_list.append(xs)
                smoothed_variance_list.append(self.H @ ps @ self.H.T + self.R)
        smoothed_state_list.append(state_list[0])
        smoothed_variance_list.append(self.H @ variance_list[0] @ self.H.T + self.R)
        return smoothed_state_list, smoothed_variance_list

    def kalman_filter(self, init_state, trajectory, plot=False, writer=None, trajectory_index=None, epoch=0, cal=None,
                      beta=0, update=True, res=None):
        self.initialize(init_state)
        EKF_res_state = [init_state.clone()]
        EKF_res_P = [self.P]
        innovation_vector = [torch.zeros(3, device=self.device)]
        innovation_variance = [self.H @ self.P @ self.H.T + self.R]
        EKF_res_collision = [False]
        i = 0
        j = 0
        length = len(trajectory)
        time_EKF = [0]
        while j < length - 1:
            i += 1
            time_EKF.append(i / 120)
            if cal is not None:
                params = cal.cal_params(self.state[0:2])
                self.system.set_params(tableDampingX=params[0], tableDampingY=params[1], tableFrictionX=params[2],
                                       tableFrictionY=params[3], restitution=params[4],
                                       rimFriction=params[5])
            self.predict(beta=beta, res=res)
            if (i - 0.5) / 120 <= trajectory[j + 1][-1] - trajectory[0][-1] <= (i + 0.5) / 120:
                self.update(trajectory[j + 1][0:3], update=update)
                j += 1
                EKF_res_state.append(self.predict_state)
                EKF_res_P.append(self.P)
                innovation_vector.append(self.y)
                innovation_variance.append(self.S)
                EKF_res_collision.append(self.has_collision)
            elif trajectory[j + 1][-1] - trajectory[0][-1] < (i - 0.5) / 120:
                j = j + 1
                i = i - 1
                self.state = self.predict_state
            else:
                self.state = self.predict_state
        if plot:
            EKF_plot_with_state_list(EKF_res_state, trajectory, writer=writer, trajectory_index=trajectory_index,
                                     epoch=epoch)
        return EKF_res_state, EKF_res_collision, innovation_vector, innovation_variance


if __name__ == '__main__':
    device = torch.device("cuda")
    # test for torch_EKF_Wrapper
    # tableDamping = 0.001
    # tableFriction = 0.001
    # tableRestitution = 0.7424
    para = [0.10608561, 0.34085548, 0.78550678]
    system = torch_air_hockey_baseline_no_detach.SystemModel(tableDamping=para[1], tableFriction=para[0],
                                                             tableLength=1.948,
                                                             tableWidth=1.038,
                                                             goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
                                                             tableRes=para[2], malletRes=0.8, rimFriction=0.1418,
                                                             dt=1 / 120, beta=30)
    R = torch.zeros((3, 3), device=device)
    R[0][0] = 2.5e-7
    R[1][1] = 2.5e-7
    R[2][2] = 9.1e-3
    Q = torch.zeros((6, 6), device=device)
    Q[0][0] = Q[1][1] = 2e-10
    Q[2][2] = Q[3][3] = 1e-7
    Q[4][4] = 1.0e-2
    Q[5][5] = 1e-1
    P = torch.eye(6, device=device) * 0.01
    pre_data = np.load("total_data.npy", allow_pickle=True)
    pre_data = pre_data[0]
    data = []
    for i in range(1, len(pre_data)):
        if abs(pre_data[i][0] - pre_data[i - 1][0]) < 0.005 and abs(pre_data[i][1] - pre_data[i - 1][1]) < 0.005:
            continue
        data.append(pre_data[i])
    for i_data in data:
        i_data[0] += system.table.m_length / 2
    data = torch.tensor(np.array(data), device=device).float()
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
    puck_EKF = AirHockeyEKF(u=1 / 120, system=system, Q=Q, R=R, P=P)
    puck_EKF.initialize(state)
    resx = [state[0]]
    resy = [state[1]]
    res_theta = [state[4]]
    time_EKF = [1 / 120]
    j = 1
    length = len(data) - 1
    i = 0
    evaluation = 0
    num_evaluation = 0
    while j < length:
        i += 1
        time_EKF.append((i + 1) / 120)
        puck_EKF.predict()
        resx.append(puck_EKF.predict_state[0])
        resy.append(puck_EKF.predict_state[1])
        res_theta.append(puck_EKF.predict_state[4])
        # check whether data is recorded at right time
        if (i - 0.2) / 120 < data[j + 1][-1] - data[1][-1] < (i + 0.2) / 120:
            puck_EKF.update(data[j + 1][0:3])
            j += 1
            sign, logdet = torch.linalg.slogdet(puck_EKF.S)
            num_evaluation += 1
            evaluation += sign * torch.exp(logdet) + puck_EKF.y @ torch.linalg.inv(puck_EKF.S) @ puck_EKF.y
        elif data[j + 1][-1] - data[1][-1] <= (i - 0.2) / 120:
            j += 1
            puck_EKF.state = puck_EKF.predict_state
        else:
            puck_EKF.state = puck_EKF.predict_state
    print(evaluation / num_evaluation)
    resx = torch.tensor(resx, device=device)
    resy = torch.tensor(resy, device=device)
    res_theta = torch.tensor(res_theta, device=device)
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
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 0].cpu().numpy(), color='g',
                label='raw data x position', s=5)
    plt.title('only raw data x position')
    plt.legend()
    plt.subplot(3, 3, 3)
    plt.scatter(time_EKF, resx.cpu().numpy(), color='b', label='EKF x position', s=5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 0].cpu().numpy(), color='g',
                label='raw data x position', s=5)
    plt.title('EKF vs raw data x position')
    plt.legend()
    plt.subplot(3, 3, 4)
    plt.scatter(time_EKF, resy.cpu().numpy(), color='b', label='EKF y position', s=5)
    plt.title('only EKF y position')
    plt.legend()
    plt.subplot(3, 3, 5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g',
                label='raw data y position', s=5)
    plt.title('only raw data y position')
    plt.legend()
    plt.subplot(3, 3, 6)
    plt.scatter(time_EKF, resy.cpu().numpy(), color='b', label='EKF y position', s=5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g',
                label='raw data y position', s=5)
    plt.title('EKF vs raw data y position')
    plt.legend()
    plt.subplot(3, 3, 7)
    plt.scatter(time_EKF, res_theta.cpu().numpy(), color='b', label='EKF theta', s=5)
    plt.title('only EKF theta')
    plt.legend()
    plt.subplot(3, 3, 8)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 2].cpu().numpy(), color='g',
                label='raw data theta', s=5)
    plt.legend()
    plt.subplot(3, 3, 9)
    plt.scatter(time_EKF, res_theta.cpu().numpy(), color='b', label='EKF theta', s=5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 2].cpu().numpy(), color='g',
                label='raw data y position', s=5)
    plt.legend()
    plt.show()
