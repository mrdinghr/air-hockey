import datetime
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

# def convert(params, middle_value):
#     return torch.sigmoid(params)*middle_value
#
# def inv_conver(params, middlevalue):


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
        # self.register_parameter('covariance_params', torch.nn.Parameter(covariance_params))
        self.covariance_params = covariance_params
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
        self.dyna_params = torch.abs(self.params)
        self.R = torch.diag(torch.exp(torch.stack([self.covariance_params[0],
                                                   self.covariance_params[1],
                                                   self.covariance_params[2]])))
        self.Q = torch.diag(torch.exp(torch.stack([self.covariance_params[3], self.covariance_params[3],
                                                   self.covariance_params[4], self.covariance_params[4],
                                                   self.covariance_params[5], self.covariance_params[6]])))
        # self.Q = torch.diag(torch.abs(torch.stack([self.covariance_params[3], self.covariance_params[3],
        #                                            self.covariance_params[4], self.covariance_params[4],
        #                                            self.covariance_params[5], self.covariance_params[6]])) + 1e-6)
        # self.R = torch.diag(torch.abs(torch.stack([self.covariance_params[0],
        #                                            self.covariance_params[1],
        #                                            self.covariance_params[2]])))
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
    def prepare_dataset(self, trajectory_buffer, writer=None, plot=False, epoch=0):
        self.construct_EKF()
        segments_dataset = []
        # Pure Kalman Smooth
        with torch.no_grad():
            for trajectory_index, trajectory in enumerate(trajectory_buffer):
                trajectory_tensor = torch.tensor(trajectory, device=self.device).float()
                init_state = self.calculate_init_state(trajectory)
                smoothed_states, smoothed_variances, collisions = self.puck_EKF.smooth(init_state, trajectory_tensor,
                                                                                       plot=plot, writer=writer,
                                                                                       epoch=epoch,
                                                                                       trajectory_index=trajectory_index)
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
        num_total_loss = 0
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
                if 0 in sign or -1 in sign:
                    print('*********************have problem**********')
                total_loss = total_loss + 0.5 * torch.sum(logdet + torch.einsum('ij, ijk, ik->i', innovation,
                                                                                           torch.linalg.inv(
                                                                                            smoothed_variance_tensor),
                                                                                           innovation))
            elif loss_type == 'mse':
                total_loss = total_loss + torch.bmm(innovation.unsqueeze(1), innovation.unsqueeze(2)).sum()
            num_total_loss += 1

        return total_loss / num_total_loss


def load_dataset(file_name):
    total_dataset = np.load(file_name, allow_pickle=True)
    # return np.array([total_dataset[3], total_dataset[3]]), total_dataset[3:4]
    return np.array([total_dataset[3][:-5], total_dataset[3][:-5]]), total_dataset[3:4]
    # plt.scatter(total_dataset[3][:, 0], total_dataset[3][:, 1])
    # plt.show()
    # return total_dataset[0:int(len(total_dataset) * 0.8)], total_dataset[int(len(total_dataset) * 0.8):]


if __name__ == '__main__':
    device = torch.device("cuda")
    file_name = 'new_total_data_after_clean.npy'
    lr = 1e-3
    batch_size = 10
    batch_trajectory_size = 10
    epochs = 1000
    beta = 1
    training_dataset, test_dataset = load_dataset(file_name)

    # table friction, table damping, table restitution, rim friction
    init_params = torch.Tensor([3e-3, 3e-3, 0.79968596, 0.10029725]).to(device=device)
    # init_params = init_params / torch.tensor([0.1, 0.1, 1, 1]) - 1
    # init_params = 0.5 * (torch.log(1 + init_params) - torch.log(1 - init_params))

    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1]).to(device=device)
    # covariance_params = torch.Tensor(
    #     [0.00118112, 0.00100000, 0.00295336, 0.00392161, 0.00100000, 0.00100000, 0.00205159]).to(device=device)
    covariance_params = torch.log(covariance_params)
    # covariance_params = torch.log(covariance_params / (1-covariance_params))
    model = Kalman_Smooth_Gradient(init_params, covariance_params, segment_size=batch_trajectory_size, device=device,
                                   beta=beta)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 0
    writer = SummaryWriter('./alldata/711test' + datetime.datetime.now().strftime("/%Y-%m-%d-%H-%M-%S"))
    for t in tqdm(range(epochs)):
        writer.add_scalar('dynamics/table damping', model.params[1], t)
        writer.add_scalar('dynamics/table friction', model.params[0], t)
        writer.add_scalar('dynamics/table restitution', model.params[2], t)
        writer.add_scalar('dynamics/rim friction', model.params[3], t)
        writer.add_scalar('covariance/R0', torch.exp(model.covariance_params[0]), t)
        writer.add_scalar('covariance/R1', torch.exp(model.covariance_params[1]), t)
        writer.add_scalar('covariance/R2', torch.exp(model.covariance_params[2]), t)
        writer.add_scalar('covariance/Q01', torch.exp(model.covariance_params[3]), t)
        writer.add_scalar('covariance/Q23', torch.exp(model.covariance_params[4]), t)
        writer.add_scalar('covariance/Q4', torch.exp(model.covariance_params[5]), t)
        writer.add_scalar('covariance/Q5', torch.exp(model.covariance_params[6]), t)

        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=True)
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.compute_loss(training_segment_dataset[index_batch], training_dataset)
            if loss.requires_grad:
                loss.backward()
                print("loss:", loss.item())
                print("dynamics: ", model.params.detach().cpu().numpy())
                optimizer.step()
                batch_loss.append(loss.detach().cpu().numpy())
            else:
                print("loss:", loss.item())
                print("dynamics: ", model.params.detach().cpu().numpy())
                batch_loss.append(loss.cpu().numpy())
            # loss.backward()
            # print("loss:", loss.item())
            # print("dynamics: ", model.params.detach().cpu().numpy())
            # optimizer.step()
            # batch_loss.append(loss.detach().cpu().numpy())
        training_loss = np.mean(batch_loss)
        writer.add_scalar('loss/training_loss', training_loss, t)
        with torch.no_grad():
            plot_trajectory(abs(model.params), training_dataset, epoch=t, writer=writer)
        test_segment_dataset = model.prepare_dataset(test_dataset)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.compute_loss(test_segment_dataset[index_batch], test_dataset)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)

    # writer.add_scalar('table damping', 0.1 * (torch.tanh(model.params[1]) + 1), t + 1)
    # writer.add_scalar('table friction', 0.1 * (torch.tanh(model.params[0]) + 1), t + 1)
    # writer.add_scalar('table restitution', (torch.tanh(model.params[2]) + 1), t + 1)
    # writer.add_scalar('rim friction', (torch.tanh(model.params[3]) + 1), t + 1)
    # writer.add_scalar('R0', torch.sigmoid(model.covariance_params[0]), t)
    # writer.add_scalar('R1', torch.sigmoid(model.covariance_params[1]), t)
    # writer.add_scalar('R2', torch.sigmoid(model.covariance_params[2]), t)
    # writer.add_scalar('Q01', torch.sigmoid(model.covariance_params[3]), t)
    # writer.add_scalar('Q23', torch.sigmoid(model.covariance_params[4]), t)
    # writer.add_scalar('Q4', torch.sigmoid(model.covariance_params[5]), t)
    # writer.add_scalar('Q5', torch.sigmoid(model.covariance_params[6]), t)
    writer.close()
