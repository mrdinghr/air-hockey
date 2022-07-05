import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torch
import torch_air_hockey_baseline_no_detach as torch_air_hockey_baseline
# import torch_air_hockey_baseline
from torch_EKF_Wrapper import AirHockeyEKF
from math import pi
from test_params import plot_with_state_list
from test_params import plot_trajectory
from tqdm import tqdm


# input: recorded trajectories
# output:init_state of this trajectory


# dyna_params: table friction, table damping, table restitution, rim friction
# input: trajectory, dyna_parameters, covariance_parameters, batch size actually is batch trajectory size
# output: state of one trajectory calculated by kalman smooth. list of tensor
def state_kalman_smooth(trajectory, in_dyna_params, covariance_params, batch_size, if_loss, beta):
    list_total_state_batch_start_point = []
    evaluation = 0
    num_evaluation = 0
    dyna_params = torch.zeros(4, device=device)
    dyna_params[0:2] = (torch.tanh(in_dyna_params[0:2].clone().detach()) + 1) * 0.1
    dyna_params[2:] = (torch.tanh(in_dyna_params[2:].clone().detach()) + 1)
    # dyna_params = in_dyna_params
    for trajectory_index in range(len(trajectory)):
        cur_trajectory = trajectory[trajectory_index]
        cur_trajectory = torch.tensor(cur_trajectory, device=device).float()
        R = torch.zeros((3, 3), device=device)
        R[0][0] = torch.sigmoid(covariance_params[0]).clone().detach()
        R[1][1] = torch.sigmoid(covariance_params[1]).clone().detach()
        R[2][2] = torch.sigmoid(covariance_params[2]).clone().detach()
        Q = torch.zeros((6, 6), device=device)
        Q[0][0] = torch.sigmoid(covariance_params[3]).clone().detach()
        Q[1][1] = torch.sigmoid(covariance_params[3]).clone().detach()
        Q[2][2] = torch.sigmoid(covariance_params[4]).clone().detach()
        Q[3][3] = torch.sigmoid(covariance_params[4]).clone().detach()
        Q[4][4] = torch.sigmoid(covariance_params[5]).clone().detach()
        Q[5][5] = torch.sigmoid(covariance_params[6]).clone().detach()
        P = torch.eye(6, device=device) * 0.01
        system = torch_air_hockey_baseline.SystemModel(tableDamping=dyna_params[1], tableFriction=dyna_params[0],
                                                       tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                       puckRadius=0.03165, malletRadius=0.04815,
                                                       tableRes=dyna_params[3],
                                                       malletRes=0.8, rimFriction=dyna_params[2], dt=1 / 120)
        table = torch_air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25,
                                                         puckRadius=0.03165, restitution=dyna_params[2],
                                                         rimFriction=dyna_params[3], dt=1 / 120, beta=beta)
        init_state = calculate_init_state(cur_trajectory)
        u = 1 / 120
        puck_EKF = AirHockeyEKF(u, system, table, Q, R, P)
        puck_EKF.initialize(init_state)
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
        time_EKF = [0]
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
            if (i - 0.1) / 120 < cur_trajectory[j + 1][-1] - cur_trajectory[1][-1] < (i + 0.1) / 120:
                puck_EKF.update(cur_trajectory[j + 1][0:3])
                j += 1
                EKF_res_update.append(True)
            elif cur_trajectory[j + 1][-1] - cur_trajectory[1][-1] <= (i - 0.1) / 120:
                j = j + 1
                puck_EKF.state = puck_EKF.predict_state
                EKF_res_update.append(0.5)
            else:
                puck_EKF.state = puck_EKF.predict_state
                EKF_res_update.append(False)
        EKF_resx = torch.tensor(EKF_resx)
        EKF_resy = torch.tensor(EKF_resy)
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
                # evaluation = evaluation + innovation@innovation
            xp = EKF_res_dynamic[-j - 1] @ EKF_res_state[-j - 2]
            if not EKF_res_collision[-j - 1]:
                if torch.sqrt(EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
                              EKF_res_state[-j - 2][3]) > 1e-6:
                    xp[2:4] = EKF_res_state[-j - 2][2:4] - u * (
                            system.tableDamping * EKF_res_state[-j - 2][2:4] + system.tableFriction * EKF_res_state[
                                                                                                          -j - 2][
                                                                                                      2:4] / torch.sqrt(
                        EKF_res_state[-j - 2][2] * EKF_res_state[-j - 2][2] + EKF_res_state[-j - 2][3] *
                        EKF_res_state[-j - 2][3]))
                else:
                    xp[2:4] = EKF_res_state[-j - 2][2:4] - u * system.tableDamping * EKF_res_state[-j - 2][2:4]
            pp = EKF_res_dynamic[-j - 1] @ EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T + Q
            c = EKF_res_P[-j - 2] @ EKF_res_dynamic[-j - 1].T @ torch.linalg.inv(pp)
            if abs(xs[4] - xp[4]) > pi:
                xp[4] = xp[4] - torch.sign(xp[4]) * 2 * pi
            if xs[5] * xp[5] < 0:
                xs[5] = -xs[5]
            xs = EKF_res_state[-j - 2] + c @ (xs - xp)
            ps = EKF_res_P[-j - 2] + c @ (ps - pp) @ c.T
            if EKF_res_update[-j - 2] or EKF_res_update[-j - 2] == 0.5:
                smooth_res_state.append(xs)
                smooth_resx.append(xs[0].cpu().numpy())
                smooth_resy.append(xs[1].cpu().numpy())
        for j in range(len(smooth_res_state)):
            if True in EKF_res_collision[j: j + batch_size]:
                if j % (batch_size / batch_size) == 0:
                    cur_state = smooth_res_state[-j - 1]
                    index_tensor = torch.tensor([trajectory_index, j + 2], device=device)
                    list_total_state_batch_start_point.append(torch.cat((index_tensor, cur_state)))
            else:
                if j % (batch_size / 2) == 0:
                    cur_state = smooth_res_state[-j - 1]
                    index_tensor = torch.tensor([trajectory_index, j + 2], device=device)
                    list_total_state_batch_start_point.append(torch.cat((index_tensor, cur_state)))
    # smooth_resx = torch.tensor(smooth_resx, device=device)
    # smooth_resy = torch.tensor(smooth_resy, device=device)
    plot_with_state_list(EKF_res_state, smooth_res_state, cur_trajectory, time_EKF)
    if if_loss:
        return evaluation / num_evaluation
    return list_total_state_batch_start_point, evaluation / num_evaluation


def convert(params, middle_value):
    return torch.sigmoid(params)*middle_value
'''
# table friction, table damping, table restitution, rim friction
# init_params = torch.Tensor([0.4*0.2159, 0.4*0.2513, 0.7936, 0.4352])
init_params = torch.Tensor([0.09881, 0.1101, 0.7515, 0.08031])
# init_params = torch.tanh(torch.tensor([0.125, 0.375, 0.675, 0.6], device=device))
# covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1])
covariance_params = torch.Tensor([0.4626, 0.4628, 0.4639, 0.4758, 0.4782, 0.4652, 0.4999])
index = 5
res, loss = state_kalman_smooth(total_trajectory_after_clean[index:index+1], init_params, covariance_params, 1, False)
plot_trajectory(index, init_params)
'''


class Kalman_Smooth_Gradient(torch.nn.Module):
    def __init__(self, params, covariance_params, beta, segment_size, device):
        super(Kalman_Smooth_Gradient, self).__init__()
        self.register_parameter('params', torch.nn.Parameter(params))
        self.register_parameter('covariance_params', torch.nn.Parameter(covariance_params))
        self.segment_size = segment_size
        self.device = device
        self.dyna_params = None
        # self.covariance_params = covariance_params
        self.system = None
        self.table = None
        self.puck_EKF = None
        self.R = None
        self.Q = None
        self.P = None
        self.beta = beta

    def construct_EKF(self):
        # self.dyna_params = (torch.tanh(self.params) + 1) * 0.5
        self.dyna_params = torch.cat([(torch.tanh(self.params[0:2]) + 1) * 0.01, (torch.tanh(self.params[2:]) + 1)])
        self.R = torch.diag(
            torch.sigmoid(torch.stack([self.covariance_params[0], self.covariance_params[1], self.covariance_params[2]])))
        self.R = self.R.to(device=self.device)
        self.Q = torch.diag(torch.sigmoid(torch.stack(
            [self.covariance_params[3], self.covariance_params[3], self.covariance_params[4], self.covariance_params[4],
             self.covariance_params[5], self.covariance_params[6]])))
        self.Q = self.Q.to(device=self.device)
        self.P = torch.eye(6, device=device) * 0.01
        self.system = torch_air_hockey_baseline.SystemModel(tableDamping=self.dyna_params[1],
                                                            tableFriction=self.dyna_params[0],
                                                            tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                            puckRadius=0.03165, malletRadius=0.04815,
                                                            tableRes=self.dyna_params[2],
                                                            malletRes=0.8, rimFriction=self.dyna_params[3], dt=1 / 120,
                                                            beta=self.beta)
        self.puck_EKF = AirHockeyEKF(u=1 / 120., system=self.system, Q=self.Q, R=self.R, P=self.P, device=self.device)

    # input batch trajectory
    def prepare_dataset(self, trajectory_buffer):
        self.construct_EKF()
        segments_dataset = []
        # Pure Kalman Smooth
        with torch.no_grad():
            for trajectory_index, trajectory in enumerate(trajectory_buffer):
                trajectory_tensor = torch.tensor(trajectory, device=self.device).float()
                init_state = self.calculate_init_state(trajectory)
                smoothed_states, smoothed_variances, collisions = self.puck_EKF.smooth(init_state, trajectory_tensor)
                segments_dataset.append(
                    torch.vstack(self.construct_data_segments(smoothed_states, collisions, trajectory_index)))
        return torch.vstack(segments_dataset)

    def calculate_init_state(self, trajectory):
        dx = ((trajectory[1][0] - trajectory[0][0]) / (trajectory[1][3] - trajectory[0][3]) + (
                trajectory[2][0] - trajectory[1][0]) / (
                      trajectory[2][3] - trajectory[1][3]) + (trajectory[3][0] - trajectory[2][0]) / (
                      trajectory[3][3] - trajectory[2][3])) / 3
        dy = ((trajectory[1][1] - trajectory[0][1]) / (trajectory[1][3] - trajectory[0][3]) + (
                trajectory[2][1] - trajectory[1][1]) / (
                      trajectory[2][3] - trajectory[1][3]) + (trajectory[3][1] - trajectory[2][1]) / (
                      trajectory[3][3] - trajectory[2][3])) / 3
        dtheta = ((trajectory[1][2] - trajectory[0][2]) / (trajectory[1][3] - trajectory[0][3]) + (
                trajectory[2][2] - trajectory[1][2]) / (
                          trajectory[2][3] - trajectory[1][3]) + (trajectory[3][2] - trajectory[2][2]) / (
                          trajectory[3][3] - trajectory[2][3])) / 3
        state_ = torch.tensor([trajectory[1][0], trajectory[1][1], dx, dy, trajectory[1][2], dtheta],
                              device=self.device).float()
        return state_

    def construct_data_segments(self, smoothed_states, collisions, trajectory_index):
        segment_dataset = []
        for j in range(len(smoothed_states) - self.segment_size + 1):
            if np.any(collisions[j: j + self.segment_size]):
                cur_state = smoothed_states[j]
                index_tensor = torch.tensor([trajectory_index, j + 2], device=device)
                segment_dataset.append(torch.cat((index_tensor, cur_state)))
            else:
                if j % (batch_size / 2) == 0:
                    cur_state = smoothed_states[j]
                    index_tensor = torch.tensor([trajectory_index, j + 2], device=device)
                    segment_dataset.append(torch.cat((index_tensor, cur_state)))
        return segment_dataset

    def compute_loss(self, smoothed_segments_batch, measurements, loss_type='log_lik'):
        self.construct_EKF()
        total_loss = 0
        for point in smoothed_segments_batch:
            trajectory_idx = int(point[0])
            trajectory_point_idx = int(point[1])
            segment_measurment = torch.tensor(
                measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size],
                device=self.device).float()
            smoothed_state_list, smoothed_variance_list, _ = self.puck_EKF.smooth(point[2:], segment_measurment)
            smoothed_state_tensor = torch.stack(smoothed_state_list)
            smoothed_variance_tensor = torch.stack(smoothed_variance_list)

            innovation_xy = segment_measurment[2:, :2] - smoothed_state_tensor[:, :2]
            innovation_angle = segment_measurment[2:, 2] - smoothed_state_tensor[:, 4]
            sign, logdet = torch.linalg.slogdet(smoothed_variance_tensor)
            idx = torch.where(segment_measurment[2:, 2] - smoothed_state_tensor[:, 4] > 3 / 2 * np.pi)[0]
            innovation_angle[idx] = innovation_angle[idx] - 2 * np.pi

            idx = torch.where(segment_measurment[2:, 2] - smoothed_state_tensor[:, 4] < -3 / 2 * np.pi)[0]
            innovation_angle[idx] = innovation_angle[idx] + 2 * np.pi

            innovation = torch.cat([innovation_xy, innovation_angle.unsqueeze(1)], dim=1)

            if loss_type == 'log_lik':
                total_loss = total_loss + torch.sum(
                    sign * torch.exp(logdet) + torch.einsum('ij, ijk, ik->i', innovation, smoothed_variance_tensor,
                                                            innovation))
            elif loss_type == 'mse':
                total_loss = total_loss + torch.bmm(innovation.unsqueeze(1), innovation.unsqueeze(2)).sum()

        return total_loss


def load_dataset(file_name):
    total_dataset = np.load(file_name, allow_pickle=True)
    return total_dataset[0:5], total_dataset[8:10]


if __name__ == '__main__':
    device = torch.device("cuda")
    file_name = 'new_total_data_after_clean.npy'
    lr = 1e-4
    batch_size = 10
    batch_trajectory_size = 10
    epochs = 150
    beta = 1
    training_dataset, test_dataset = load_dataset(file_name)

    # table friction, table damping, table restitution, rim friction
    init_params = torch.Tensor([0.1, 0.15, 0.8, 0.10])
    init_params = init_params / torch.tensor([0.1, 0.1, 1, 1]) - 1
    init_params = 0.5 * (torch.log(1 + init_params) - torch.log(1 - init_params))

    # covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1])
    covariance_params = torch.Tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    covariance_params = torch.log(covariance_params / (1-covariance_params))
    model = Kalman_Smooth_Gradient(init_params, covariance_params, segment_size=batch_trajectory_size, device=device,
                                   beta=beta)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch = 0
    writer = SummaryWriter('./smoothcov74')
    for t in tqdm(range(epochs)):
        writer.add_scalar('table damping', 0.1 * (torch.tanh(model.params[1]) + 1), t)
        writer.add_scalar('table friction', 0.1 * (torch.tanh(model.params[0]) + 1), t)
        writer.add_scalar('table restitution', (torch.tanh(model.params[2]) + 1), t)
        writer.add_scalar('rim friction', (torch.tanh(model.params[3]) + 1), t)
        writer.add_scalar('R0', torch.sigmoid(model.covariance_params[0]), t)
        writer.add_scalar('R1', torch.sigmoid(model.covariance_params[1]), t)
        writer.add_scalar('R2', torch.sigmoid(model.covariance_params[2]), t)
        writer.add_scalar('Q01', torch.sigmoid(model.covariance_params[3]), t)
        writer.add_scalar('Q23', torch.sigmoid(model.covariance_params[4]), t)
        writer.add_scalar('Q4', torch.sigmoid(model.covariance_params[5]), t)
        writer.add_scalar('Q5', torch.sigmoid(model.covariance_params[6]), t)

        training_segment_dataset = model.prepare_dataset(training_dataset)
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)

        batch_loss = []
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.compute_loss(training_segment_dataset[index_batch], training_dataset)
            loss.backward()
            print(model.params.grad)
            optimizer.step()
            batch_loss.append(loss.detach().cpu().numpy())

        training_loss = np.mean(batch_loss)
        writer.add_scalar('training_loss', training_loss, t)

        test_segment_dataset = model.prepare_dataset(test_dataset)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.compute_loss(test_segment_dataset[index_batch], test_dataset)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('test_loss', test_loss, t)

    writer.add_scalar('table damping', 0.1 * (torch.tanh(model.params[1]) + 1), t + 1)
    writer.add_scalar('table friction', 0.1 * (torch.tanh(model.params[0]) + 1), t + 1)
    writer.add_scalar('table restitution', (torch.tanh(model.params[2]) + 1), t + 1)
    writer.add_scalar('rim friction', (torch.tanh(model.params[3]) + 1), t + 1)
    writer.add_scalar('R0', torch.sigmoid(model.covariance_params[0]), t)
    writer.add_scalar('R1', torch.sigmoid(model.covariance_params[1]), t)
    writer.add_scalar('R2', torch.sigmoid(model.covariance_params[2]), t)
    writer.add_scalar('Q01', torch.sigmoid(model.covariance_params[3]), t)
    writer.add_scalar('Q23', torch.sigmoid(model.covariance_params[4]), t)
    writer.add_scalar('Q4', torch.sigmoid(model.covariance_params[5]), t)
    writer.add_scalar('Q5', torch.sigmoid(model.covariance_params[6]), t)
    writer.close()
