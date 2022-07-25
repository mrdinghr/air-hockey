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


class StateDependentParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 6)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc4.weight, gain=1.0)

    def cal_params(self, state):
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc4(output)
        # output = torch.nn.functional.relu(output)
        output = torch.sigmoid(output)
        return output


class ResState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 6)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc4.weight, gain=1.0)

    def cal_res(self, state):
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc4(output)
        # output = torch.nn.functional.relu(output)
        output = torch.sigmoid(output)
        return output


class FixedParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("dyna_params", torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))

    def cal_params(self, state):
        if len(state.size()) == 1:
            return torch.sigmoid(self.dyna_params)
        elif len(state.size()) == 2:
            return torch.sigmoid(self.dyna_params.tile(state.shape[-2], 1))


if __name__ == '__main__':
    file_name = 'new_total_data_after_clean_part.npy'
    training_dataset, test_dataset = load_dataset(file_name)
    torch.manual_seed(0)
    device = torch.device("cuda")
    lr = 1e-4
    batch_size = 2
    batch_trajectory_size = 10
    epochs = 2000
    # cal = StateDependentParams()
    cal = FixedParams()
    cal.to(device)
    res = ResState()
    res.to(device)
    # cal.load_state_dict(torch.load('./alldata/718nn/2022-07-22-10-38-29smsmonecollbigcov/model.pt'))
    # params: damping x, damping y, friction x, friction y, restitution, rimfriction
    init_params = cal.cal_params(torch.tensor([0., 0.], device=device))
    #  R0， R1， R2， Q01， Q23，Q4， Q5
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1]).to(device=device)
    covariance_params = torch.log(covariance_params)
    set_params = False
    set_res = True
    model = Kalman_EKF_Gradient(init_params, covariance_params, segment_size=batch_trajectory_size, device=device,
                                set_params=set_params)
    optimizer = torch.optim.Adam(cal.parameters(), lr=lr)
    epoch = 0
    logdir = './alldata/718nn' + datetime.datetime.now().strftime("/%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(logdir)
    prepare_typ = 'smooth'
    loss_typ = 'smooth'
    for t in tqdm(range(epochs)):
        # params: damping x, damping y, friction x, friction y, restitution, rimfriction
        beta = 29 * t / epochs + 1
        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=True,
                                                         type=prepare_typ, set_params=set_params, cal=cal,
                                                         beta=beta, set_res=set_res, res=res)
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []

        params = cal.cal_params(training_segment_dataset[:, 2:4]).mean(dim=0)
        writer.add_scalar('dynamics/table damping x', params[0], t)
        writer.add_scalar('dynamics/table damping y', params[1], t)
        writer.add_scalar('dynamics/table friction x', params[2], t)
        writer.add_scalar('dynamics/table friction y', params[3], t)
        writer.add_scalar('dynamics/table restitution', params[4], t)
        writer.add_scalar('dynamics/rim friction', params[5], t)
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(training_segment_dataset[index_batch], training_dataset, type=loss_typ, epoch=t,
                                        set_params=set_params, cal=cal, beta=beta, set_res=set_res, res=res)
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
            plot_trajectory(abs(model.params), training_dataset, epoch=t, writer=writer, set_params=True, cal=cal,
                            beta=beta, set_res=set_res, res=res)
        test_segment_dataset = model.prepare_dataset(test_dataset, type=prepare_typ, epoch=t, set_params=set_params,
                                                     cal=cal, beta=beta, set_res=set_res, res=res)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.calculate_loss(test_segment_dataset[index_batch], test_dataset, type=loss_typ, epoch=t,
                                            set_params=True, cal=cal, beta=beta, set_res=set_res, res=res)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)

        if t % 50 == 0:
            torch.save(cal.state_dict(), logdir + "/model.pt")
