import datetime
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_air_hockey_baseline_no_detach as torch_air_hockey_baseline
from torch_EKF_Wrapper import AirHockeyEKF
from math import pi
from test_params import plot_trajectory
from tqdm import tqdm
torch.set_printoptions(threshold=torch.inf)


class Kalman_EKF_Gradient(torch.nn.Module):
    def __init__(self, params, covariance_params, beta, segment_size, device):
        super(Kalman_EKF_Gradient, self).__init__()
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
        R = torch.diag(torch.exp(torch.stack([self.covariance_params[0],
                                                   self.covariance_params[1],
                                                   self.covariance_params[2]])))
        Q = torch.diag(torch.exp(torch.stack([self.covariance_params[3], self.covariance_params[3],
                                                   self.covariance_params[4], self.covariance_params[4],
                                                   self.covariance_params[5], self.covariance_params[6]])))
        # self.Q = torch.diag(torch.abs(torch.stack([self.covariance_params[3], self.covariance_params[3],
        #                                            self.covariance_params[4], self.covariance_params[4],
        #                                            self.covariance_params[5], self.covariance_params[6]])) + 1e-6)
        # self.R = torch.diag(torch.abs(torch.stack([self.covariance_params[0],
        #                                            self.covariance_params[1],
        #                                            self.covariance_params[2]])))
        P = torch.eye(6, device=device) * 0.01
        self.system = torch_air_hockey_baseline.SystemModel(tableDamping=self.dyna_params[1],
                                                            tableFriction=self.dyna_params[0],
                                                            tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                            puckRadius=0.03165, malletRadius=0.04815,
                                                            tableRes=self.dyna_params[2],
                                                            malletRes=0.8, rimFriction=self.dyna_params[3], dt=1 / 120,
                                                            beta=self.beta)
        self.puck_EKF = AirHockeyEKF(u=1 / 120., system=self.system, Q=Q, R=R, P=P, device=self.device)

    def prepare_dataset(self, trajectory_buffer, writer=None, plot=False, epoch=0, type='EKF'):
        if type == 'EKF':
            self.construct_EKF()
            segments_dataset = []
            with torch.no_grad():
                for trajectory_index, trajectory in enumerate(trajectory_buffer):
                    trajectory_tensor = torch.tensor(trajectory, device=self.device).float()
                    init_state = self.calculate_init_state(trajectory)
                    EKF_state, collisions, _, _ = self.puck_EKF.kalman_filter(init_state, trajectory_tensor[1:],
                                                                              plot=plot,
                                                                              writer=writer,
                                                                              trajectory_index=trajectory_index,
                                                                              epoch=epoch)
                    segment = self.construct_data_segments(EKF_state, collisions, trajectory_index)
                    segments_dataset.append(
                        torch.vstack(segment))
            return torch.vstack(segments_dataset)
        if type == 'smooth':
            self.construct_EKF()
            segments_dataset = []
            # Pure Kalman Smooth
            with torch.no_grad():
                for trajectory_index, trajectory in enumerate(trajectory_buffer):
                    trajectory_tensor = torch.tensor(trajectory, device=self.device).float()
                    init_state = self.calculate_init_state(trajectory)
                    smoothed_states, smoothed_variances, collisions = self.puck_EKF.smooth(init_state,
                                                                                           trajectory_tensor[1:],
                                                                                           plot=plot, writer=writer,
                                                                                           epoch=epoch,
                                                                                           trajectory_index=trajectory_index)
                    segments_dataset.append(
                        torch.vstack(
                            self.construct_data_segments(smoothed_states, collisions, trajectory_index)))
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

    def construct_data_segments(self, EKF_state, collisions, trajectory_index):
        segment_dataset = []
        for j in range(len(EKF_state) - self.segment_size + 1):
            if np.any(collisions[j: j + self.segment_size]):
                cur_state = EKF_state[j]
                index_tensor = torch.tensor([trajectory_index, j + 1], device=device)
                segment_dataset.append(torch.cat((index_tensor, cur_state)))
            else:
                if j % (batch_size / 2) == 0:
                    cur_state = EKF_state[j]
                    index_tensor = torch.tensor([trajectory_index, j + 1], device=device)
                    segment_dataset.append(torch.cat((index_tensor, cur_state)))
        return segment_dataset

    def calculate_loss(self, segments_batch, measurements, loss_type='log_like', type='EKF', epoch=0):
        if type == 'EKF':
            self.construct_EKF()
            total_loss = 0
            num_total_loss = 0
            for point in segments_batch:
                trajectory_idx = int(point[0])
                trajectory_point_idx = int(point[1])
                segment_measurment = torch.tensor(
                    measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size],
                    device=self.device).float()
                EKF_state, _, innovation_vectors, innovation_variance = self.puck_EKF.kalman_filter(point[2:],
                                                                                                    segment_measurment,
                                                                                                    epoch=epoch)
                EKF_state = torch.stack(EKF_state)
                # plt.figure()
                # plt.subplot(3, 1, 1)
                # plt.title('x position')
                # plt.scatter(measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 3],
                #             measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 0],
                #             c='g')
                # plt.scatter(
                #     measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 3],
                #     EKF_state[:, 0].detach().cpu(), c='b')
                # plt.legend
                # plt.subplot(3, 1, 2)
                # plt.title('y position')
                # plt.scatter(measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 3],
                #             measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 1],
                #             c='g')
                # plt.scatter(
                #     measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 3],
                #     EKF_state[:, 1].detach().cpu(), c='b')
                # plt.legend()
                # plt.subplot(3, 1, 3)
                # plt.scatter(measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 3],
                #             measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 2],
                #             c='g')
                # plt.scatter(
                #     measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size, 3],
                #     EKF_state[:, 4].detach().cpu(), c='b')
                # plt.legend()
                # plt.show()
                for i in range(len(innovation_vectors)):
                    sign, logdet = torch.linalg.slogdet(innovation_variance[i])
                    if innovation_vectors[i][2] > 1.5 * pi:
                        innovation_vectors2 = innovation_vectors[i][2] - 2 * pi
                    elif innovation_vectors[i][2] < -1.5 * pi:
                        innovation_vectors2 = innovation_vectors[i][2] + 2 * pi
                    else:
                        innovation_vectors2 = innovation_vectors[i][2]
                    innovation = torch.cat([innovation_vectors[i][0:2], torch.atleast_1d(innovation_vectors2)])
                    if loss_type == 'log_like':
                        total_loss = total_loss + 0.5 * (
                                logdet + innovation @ innovation_variance[i].inverse() @ innovation)
                    elif loss_type == 'mse':
                        total_loss = total_loss + 0.5 * (innovation @ innovation)
                num_total_loss += 1
            return total_loss / num_total_loss
        if type == 'smooth':
            self.construct_EKF()
            total_loss = 0
            num_total_loss = 0
            for point in segments_batch:
                trajectory_idx = int(point[0])
                trajectory_point_idx = int(point[1])
                segment_measurment = torch.tensor(
                    measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size],
                    device=self.device).float()
                smoothed_state_list, smoothed_variance_list, _ = self.puck_EKF.smooth(point[2:], segment_measurment,
                                                                                      epoch=epoch)
                smoothed_state_tensor = torch.stack(smoothed_state_list)
                smoothed_variance_tensor = torch.stack(smoothed_variance_list)

                innovation_xy = segment_measurment[:, :2] - smoothed_state_tensor[:, :2]
                innovation_angle = segment_measurment[:, 2] - smoothed_state_tensor[:, 4]
                sign, logdet = torch.linalg.slogdet(smoothed_variance_tensor)
                idx = torch.where(segment_measurment[:, 2] - smoothed_state_tensor[:, 4] > 3 / 2 * np.pi)[0]
                innovation_angle[idx] = innovation_angle[idx] - 2 * np.pi

                idx = torch.where(segment_measurment[:, 2] - smoothed_state_tensor[:, 4] < -3 / 2 * np.pi)[0]
                innovation_angle[idx] = innovation_angle[idx] + 2 * np.pi

                innovation = torch.cat([innovation_xy, innovation_angle.unsqueeze(1)], dim=1)
                if loss_type == 'log_like':
                    if 0 in sign or -1 in sign:
                        print('*********************have problem**********')
                    loss_i = torch.einsum('ij, ijk, ik->i', innovation, torch.linalg.inv(smoothed_variance_tensor),innovation)
                    print("loss_i: ", loss_i)
                    total_loss = total_loss + 0.5 * torch.sum(logdet + loss_i)
                elif loss_type == 'mse':
                    total_loss = total_loss + torch.bmm(innovation.unsqueeze(1), innovation.unsqueeze(2)).sum()
                num_total_loss += 1

            return total_loss / num_total_loss


def load_dataset(file_name):
    total_dataset = np.load(file_name, allow_pickle=True)
    # total_dataset[2] = total_dataset[2][50:]
    return total_dataset[2:3], total_dataset[2:3]
    # return total_dataset[2:3], total_dataset[2:3]
    '''
    'new_total_data_after_clean.npy' forth trajectory only one collision on down wall np.array([total_dataset[3][:-5], total_dataset[3][:-5]])
    'new_total_data_after_clean.npy' first trajectory one collision on up wall  np.array([total_dataset[0][25:120], total_dataset[0][25:120]])
    'new_total_data_after_clean.npy' first trajectory one collision on left wall  np.array([total_dataset[5][40:95], total_dataset[5][40:95]])
    'new_total_data_after_clean.npy' first trajectory one collision on right wall  np.array([total_dataset[5][250:320], total_dataset[5][250:320]])
    '''
    # plt.scatter(total_dataset[3][:, 0], total_dataset[3][:, 1])
    # plt.show()
    # return total_dataset[0:int(len(total_dataset) * 0.8)], total_dataset[int(len(total_dataset) * 0.8):]


if __name__ == '__main__':
    device = torch.device("cuda")
    file_name = 'new_total_data_after_clean_part.npy'
    torch.manual_seed(0)
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
    #                                      R0, R1, R2, Q01, Q23, Q4, Q5
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1]).to(device=device)
    # covariance_params = torch.Tensor(
    #     [0.00118112, 0.00100000, 0.00295336, 0.00392161, 0.00100000, 0.00100000, 0.00205159]).to(device=device)
    covariance_params = torch.log(covariance_params)
    # covariance_params = torch.log(covariance_params / (1-covariance_params))
    model = Kalman_EKF_Gradient(init_params, covariance_params, segment_size=batch_trajectory_size, device=device,
                                beta=beta)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch = 0
    writer = SummaryWriter('./alldata/712test' + datetime.datetime.now().strftime("/%Y-%m-%d-%H-%M-%S"))
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
        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=True,
                                                         type='smooth')
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(training_segment_dataset[index_batch], training_dataset, type='EKF', epoch=t)
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
        test_segment_dataset = model.prepare_dataset(test_dataset, type='smooth', epoch=t)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.calculate_loss(test_segment_dataset[index_batch], test_dataset, type='EKF', epoch=t)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)
    writer.close()
