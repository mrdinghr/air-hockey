import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_air_hockey_baseline_no_detach as torch_air_hockey_baseline
from torch_EKF_Wrapper import AirHockeyEKF
from math import pi
from test_params import plot_trajectory
from tqdm import tqdm
from torch_gradient import Kalman_EKF_Gradient
from torch_gradient import load_dataset

torch.set_printoptions(threshold=torch.inf)


class state_dependent_params(torch.nn.Module):
    def __init__(self):
        super(state_dependent_params, self).__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 4)

    def cal_params(self, state):
        params = self.fc1(state)
        params = torch.nn.functional.relu(params)
        params = self.fc2(params)
        params = torch.nn.functional.relu(params)
        params = self.fc3(params)
        params = torch.nn.functional.relu(params)
        params = torch.sigmoid(params)
        return params


if __name__ == '__main__':
    file_name = 'new_total_data_no_collision.npy'
    training_dataset, test_dataset = load_dataset(file_name)
    torch.manual_seed(0)
    device = torch.device("cuda")
    lr = 1e-3
    batch_size = 10
    batch_trajectory_size = 10
    epochs = 200
    cal = state_dependent_params()
    cal.to(device)
    init_params = cal.cal_params(torch.tensor([0., 0., 0.], device=device))
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1]).to(device=device)
    model = Kalman_EKF_Gradient(init_params, covariance_params, segment_size=batch_trajectory_size, device=device)
    optimizer = torch.optim.Adam(cal.parameters(), lr=lr)
    epoch = 0
    writer = SummaryWriter('./alldata/717nn' + datetime.datetime.now().strftime("/%Y-%m-%d-%H-%M-%S"))
    for t in tqdm(range(epochs)):
        params = cal.cal_params(torch.tensor([0., 0., 0.], device=device))
        writer.add_scalar('dynamics/table damping', params[1], t)
        writer.add_scalar('dynamics/table friction', params[0], t)
        writer.add_scalar('dynamics/table restitution', params[2], t)
        writer.add_scalar('dynamics/rim friction', params[3], t)
        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=True,
                                                         type='smooth', set_params=True, cal=cal)
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(training_segment_dataset[index_batch], training_dataset, type='smooth', epoch=t,
                                        set_params=True, cal=cal)
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
        training_loss = np.mean(batch_loss)
        writer.add_scalar('loss/training_loss', training_loss, t)
        with torch.no_grad():
            plot_trajectory(abs(model.params), training_dataset, epoch=t, writer=writer, set_params=True, cal=cal)
        test_segment_dataset = model.prepare_dataset(test_dataset, type='smooth', epoch=t, set_params=True, cal=cal)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.calculate_loss(test_segment_dataset[index_batch], test_dataset, type='smooth', epoch=t,
                                            set_params=True, cal=cal)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)
