import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torch
import torch_air_hockey_baseline_no_detach as torch_air_hockey_baseline
# import torch_air_hockey_baseline
from torch_EKF_Wrapper import air_hockey_EKF
from math import pi

device = torch.device("cuda")
table_length = 1.948
total_trajectory_after_clean = np.load('total_data_after_clean.npy', allow_pickle=True)
train_trajectory = total_trajectory_after_clean[65:70]
test_trajectory = total_trajectory_after_clean[10:12]


# input: recorded trajectories
# output:init_state of this trajectory
def calculate_init_state(data):
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
    return state


# dyna_params: table friction, table damping, table restitution, rim friction
# input: trajectory, dyna_parameters, covariance_parameters
# output: state of one trajectory calculated by kalman smooth. list of tensor
def state_kalman_smooth(trajectory, in_dyna_params, covariance_params, batch_size, if_loss):
    list_total_state_batch_start_point = []
    evaluation = 0
    num_evaluation = 0
    for trajectory_index in range(len(trajectory)):
        cur_trajectory = trajectory[trajectory_index]
        cur_trajectory = torch.tensor(cur_trajectory, device=device).float()
        dyna_params = in_dyna_params.clone().detach()
        R = torch.zeros((3, 3), device=device)
        R[0][0] = covariance_params[0]
        R[1][1] = covariance_params[1]
        R[2][2] = covariance_params[2]
        Q = torch.zeros((6, 6), device=device)
        Q[0][0] = covariance_params[3]
        Q[1][1] = covariance_params[3]
        Q[2][2] = covariance_params[4]
        Q[3][3] = covariance_params[4]
        Q[4][4] = covariance_params[5]
        Q[5][5] = covariance_params[6]
        P = torch.eye(6, device=device) * 0.01
        system = torch_air_hockey_baseline.SystemModel(tableDamping=dyna_params[1], tableFriction=dyna_params[0],
                                                       tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                       puckRadius=0.03165, malletRadius=0.04815,
                                                       tableRes=dyna_params[3],
                                                       malletRes=0.8, rimFriction=dyna_params[2], dt=1 / 120)
        table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25,
                                                         puckRadius=0.03165, restitution=dyna_params[2],
                                                         rimFriction=dyna_params[3], dt=1 / 120)
        init_state = calculate_init_state(cur_trajectory)
        u = 1 / 120
        puck_EKF = air_hockey_EKF(u, system, table, Q, R, P)
        puck_EKF.init_state(init_state)
        EKF_res_state = [init_state]
        EKF_resx = [init_state[0]]
        EKF_resy = [init_state[1]]
        EKF_res_P = []
        EKF_res_dynamic = []
        EKF_res_collision = []
        EKF_res_update = []
        i = 0
        j = 1
        length = len(cur_trajectory)
        time_EKF = []
        while j < length - 1:
            i += 1
            time_EKF.append(i / 120)
            puck_EKF.predict()
            EKF_res_state.append(puck_EKF.predict_state)
            EKF_res_P.append(puck_EKF.P)
            EKF_res_dynamic.append(puck_EKF.F)
            EKF_res_collision.append(puck_EKF.has_collision)
            EKF_resx.append(puck_EKF.predict_state[0])
            EKF_resy.append(puck_EKF.predict_state[1])
            if (i - 0.2) / 120 < cur_trajectory[j + 1][-1] - cur_trajectory[1][-1] < (i + 0.2) / 120:
                puck_EKF.update(cur_trajectory[j + 1][0:3])
                j += 1
                EKF_res_update.append(True)
            elif cur_trajectory[j + 1][-1] - cur_trajectory[1][-1] <= (i - 0.2) / 120:
                j = j + 1
                puck_EKF.state = puck_EKF.predict_state
                EKF_res_update.append(False)
            else:
                puck_EKF.state = puck_EKF.predict_state
                EKF_res_update.append(False)
        smooth_res_state = [EKF_res_state[-1]]
        smooth_resx = [EKF_resx[-1]]
        smooth_resy = [EKF_resy[-1]]
        xs = EKF_res_state[-1]
        ps = EKF_res_P[-1]
        time = len(EKF_res_state)
        i = 0
        for j in range(time - 2):
            if EKF_res_update[-1 - j]:
                i += 1
                innovation = cur_trajectory[-i, 0:3] - torch.tensor([xs[0], xs[1], xs[4]], device=device)
                if xs[4] * cur_trajectory[-i, 2] < 0:
                    innovation[2] = 2 * pi + torch.sign(xs[4]) * (cur_trajectory[-i, 2] - xs[4])
                innovation_covariance = puck_EKF.H @ ps @ puck_EKF.H.T + puck_EKF.R
                sign, logdet = torch.linalg.slogdet(innovation_covariance)
                num_evaluation += 1
                evaluation = evaluation + (sign * torch.exp(logdet) + innovation @ torch.linalg.inv(
                    innovation_covariance) @ innovation)
            xp = EKF_res_dynamic[-j - 1] @ EKF_res_state[-j - 2]
            # if not EKF_res_collision[-j - 1]:
            #     if torch.sqrt(EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
            #                EKF_res_state[-j - 2][3]) > 1e-6:
            #         xp[2:4] = EKF_res_state[-j - 2][2:4] - u * (
            #                 system.tableDamping * EKF_res_state[-j - 2][2:4] + system.tableFriction * EKF_res_state[-j - 2][
            #                                                                                           2:4] / torch.sqrt(
            #             EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
            #             EKF_res_state[-j - 2][3]))
            #     else:
            #         xp[2:4] = EKF_res_state[-j - 2][2:4] - u * system.tableDamping * EKF_res_state[-j - 2][2:4]
            pp = EKF_res_dynamic[-j - 1] @ EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T + Q
            c = EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T @ torch.linalg.inv(pp)
            if abs(xs[4] - xp[4]) > pi:
                xp[4] = xp[4] - torch.sign(xp[4]) * 2 * pi
            if xs[5] * xp[5] < 0:
                xs[5] = -xs[5]
            xs = EKF_res_state[-j - 2] + c @ (xs - xp)
            ps = EKF_res_P[-j - 2] + c @ (ps - pp) @ c.T
            if EKF_res_update[-j - 2]:
                smooth_res_state.append(xs)
                smooth_resx.append(xs[0].cpu().numpy())
                smooth_resy.append(xs[1].cpu().numpy())
        for j in range(len(smooth_res_state) + 1):
            if j % batch_size == 0:
                cur_state = smooth_res_state[-1]
                index_tensor = torch.tensor([trajectory_index, j + 2], device=device)
                list_total_state_batch_start_point.append(torch.cat((index_tensor, cur_state)))
    # EKF_resx = torch.tensor(EKF_resx, device=device)
    # EKF_resy = torch.tensor(EKF_resy, device=device)
    # smooth_resx = torch.tensor(smooth_resx, device=device)
    # smooth_resy = torch.tensor(smooth_resy, device=device)
    # plt.figure()
    # plt.scatter(cur_trajectory[1:, 0].cpu().numpy(), cur_trajectory[1:, 1].cpu().numpy(), label='recorded trajectory', alpha=0.5)
    # # plt.scatter(EKF_resx.cpu().numpy(), EKF_resy.cpu().numpy(), label='EKF trajectory', alpha=0.5)
    # plt.scatter(smooth_resx.cpu().numpy(), smooth_resy.cpu().numpy(), label='Smooth trajectory')
    # plt.legend()
    # plt.show()
    if if_loss:
        return evaluation / num_evaluation
    return list_total_state_batch_start_point, evaluation / num_evaluation


# init_params = torch.Tensor([0.125, 0.375, 0.675, 0.145])
# init_params.requires_grad = True
# covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1])
# res = state_kalman_smooth(total_trajectory_after_clean[0:5], init_params, covariance_params, 20)
# print(res)


class Kalman_Smooth_Gradient(torch.nn.Module):
    def __init__(self, params, covariance_params):
        super(Kalman_Smooth_Gradient, self).__init__()
        self.register_parameter('dyna_params', torch.nn.Parameter(params))
        self.covariance_params = covariance_params
        self.system = torch_air_hockey_baseline.SystemModel(tableDamping=self.dyna_params[1],
                                                            tableFriction=self.dyna_params[0],
                                                            tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                            puckRadius=0.03165, malletRadius=0.04815,
                                                            tableRes=self.dyna_params[3],
                                                            malletRes=0.8, rimFriction=self.dyna_params[2], dt=1 / 120)
        self.table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25,
                                                              puckRadius=0.03165,
                                                              restitution=self.dyna_params[2],
                                                              rimFriction=self.dyna_params[3], dt=1 / 120)
        self.R = torch.zeros((3, 3), device=device)
        self.R[0][0] = self.covariance_params[0]
        self.R[1][1] = self.covariance_params[1]
        self.R[2][2] = self.covariance_params[2]
        self.Q = torch.zeros((6, 6), device=device)
        self.Q[0][0] = self.covariance_params[3]
        self.Q[1][1] = self.covariance_params[3]
        self.Q[2][2] = self.covariance_params[4]
        self.Q[3][3] = self.covariance_params[4]
        self.Q[4][4] = self.covariance_params[5]
        self.Q[5][5] = self.covariance_params[6]
        self.P = torch.eye(6, device=device) * 0.01
        self.puck_EKF = air_hockey_EKF(u=1 / 120., system=self.system, table=self.table, Q=self.Q, R=self.R, P=self.P)

    # input batch trajectory and
    def loss_kalman_smooth(self, state_list, list_index, batch_size, train_trajectory):
        evaluation = 0
        num_evaluation = 0
        for batch_index in list_index:
            trajectory_index = state_list[batch_index][0]
            start_point_index = state_list[batch_index][1]
            batch_trajectory = torch.tensor(
                train_trajectory[int(trajectory_index)][int(start_point_index):int(start_point_index) + batch_size],
                device=device).float()
            if len(batch_trajectory) <= 2:
                continue
            state = state_list[batch_index][2:]
            EKF_res_state = [state]
            EKF_res_P = []
            EKF_res_dynamic = []
            EKF_res_collision = []
            EKF_res_update = []
            self.puck_EKF.refresh(self.P, self.Q, self.R)
            i = 0
            j = 1
            self.puck_EKF.init_state(state)
            length = len(batch_trajectory)
            while j < length - 1:
                i += 1
                self.puck_EKF.predict()
                EKF_res_state.append(self.puck_EKF.predict_state)
                EKF_res_P.append(self.puck_EKF.P)
                EKF_res_dynamic.append(self.puck_EKF.F)
                EKF_res_collision.append(self.puck_EKF.has_collision)
                if (i - 0.2) / 120 < batch_trajectory[j + 1][-1] - batch_trajectory[1][-1] < (i + 0.2) / 120:
                    self.puck_EKF.update(batch_trajectory[j + 1][0:3])
                    j += 1
                    EKF_res_update.append(True)
                elif batch_trajectory[j + 1][-1] - batch_trajectory[1][-1] <= (i - 0.2) / 120:
                    j = j + 1
                    self.puck_EKF.state = self.puck_EKF.predict_state
                    EKF_res_update.append(False)
                else:
                    self.puck_EKF.state = self.puck_EKF.predict_state
                    EKF_res_update.append(False)
            time = len(EKF_res_state)
            i = 0
            xs = EKF_res_state[-1]
            ps = EKF_res_P[-1]
            # kalman smooth
            for j in range(time - 2):
                if EKF_res_update[-1 - j]:
                    i += 1
                    innovation = torch.zeros(3, device=device)
                    innovation[0] = batch_trajectory[-i, 0] - xs[0]
                    innovation[1] = batch_trajectory[-i, 1] - xs[1]
                    innovation[2] = batch_trajectory[-i, 2] - xs[4]
                    # innovation = batch_trajectory[-i, 0:3] - torch.tensor([xs[0], xs[1], xs[4]], device=device)
                    if xs[4] * batch_trajectory[-i, 2] < 0:
                        innovation[2] = 2 * pi + torch.sign(xs[4]) * (batch_trajectory[-i, 2] - xs[4])
                    innovation_covariance = self.puck_EKF.H @ ps @ self.puck_EKF.H.T + self.puck_EKF.R
                    sign, logdet = torch.linalg.slogdet(innovation_covariance)
                    num_evaluation += 1
                    evaluation = evaluation + (sign * torch.exp(logdet) + innovation @ torch.linalg.inv(
                        innovation_covariance) @ innovation)
                xp = EKF_res_dynamic[-j - 1] @ EKF_res_state[-j - 2]
                pp = EKF_res_dynamic[-j - 1] @ EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T + self.Q
                c = EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T @ torch.linalg.inv(pp)
                if abs(xs[4] - xp[4]) > pi:
                    xp[4] = xp[4] - torch.sign(xp[4]) * 2 * pi
                if xs[5] * xp[5] < 0:
                    xs[5] = -xs[5]
                xs = EKF_res_state[-j - 2] + c @ (xs - xp)
                ps = EKF_res_P[-j - 2] + c @ (ps - pp) @ c.T
        return evaluation / num_evaluation


if __name__ == '__main__':
    # table friction, table damping, table restitution, rim friction
    init_params = torch.Tensor([0.125, 0.375, 0.675, 0.145])
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1])
    model = Kalman_Smooth_Gradient(init_params, covariance_params)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Batch_size = 50
    epoch = 0
    writer = SummaryWriter('./smooth')
    for t in range(100):
        state_list, train_loss = state_kalman_smooth(train_trajectory, model.dyna_params, covariance_params, Batch_size, False)
        index_list = range(len(state_list))
        loader = Data.DataLoader(index_list, batch_size=Batch_size, shuffle=True)
        for index_batch in loader:
            optimizer.zero_grad()
            loss = model.loss_kalman_smooth(state_list, index_batch, Batch_size, train_trajectory)
            loss.backward()
            print(str(epoch)+' loss ' + str(loss))
            print('params ' + str(model.get_parameter('dyna_params').data))
            print('grad ' + str(model.get_parameter('dyna_params').grad))
            optimizer.step()
            epoch += 1
        test_loss = state_kalman_smooth(test_trajectory, model.dyna_params, covariance_params, Batch_size, True)
        writer.add_scalar('loss of train set', train_loss, t)
        writer.add_scalar('loss of test set', test_loss, t)
        writer.add_scalar('table damping', model.dyna_params[1], t)
        writer.add_scalar('table friction', model.dyna_params[0], t)
        writer.add_scalar('table restitution', model.dyna_params[2], t)
        writer.add_scalar('rim friction', model.dyna_params[3], t)
        print('test loss ' + str(test_loss))
