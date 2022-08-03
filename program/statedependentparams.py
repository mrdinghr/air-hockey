import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_air_hockey_baseline_no_detach as torch_air_hockey_baseline
from torch_EKF_Wrapper import AirHockeyEKF
from math import pi
# from test_params import plot_trajectory
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
        self.fc5 = torch.nn.Linear(64, 6)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc4.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc5.weight, gain=1.0)

    def cal_res(self, state):
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc4(output)
        output = torch.tensor([0.0, 0.0, 0.1, 0.1, 0.0, 1], device=device) * torch.tanh(output)
        # output = torch.nn.functional.relu(output)

        return output

    def cal_res_collision(self, state):
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc5(output)
        output = torch.tensor([0.0, 0.0, 1., 1., 0.0, 30], device=device) * torch.tanh(output)
        return output

class FixedParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("dyna_params", torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.05, 0.05, 0.8, 0.15])))

    def cal_params(self, state):
        if len(state.size()) == 1:
            return torch.abs(self.dyna_params)
        elif len(state.size()) == 2:
            return torch.abs(self.dyna_params.tile(state.shape[-2], 1))


if __name__ == '__main__':
    file_name = 'new_total_data_after_clean_part.npy'
    training_dataset, test_dataset = load_dataset(file_name)
    torch.manual_seed(0)
    device = torch.device("cuda")
    lr = 1e-5
    batch_size = 2
    batch_trajectory_size = 30
    epochs = 2000
    # cal = StateDependentParams()
    # cal = FixedParams()
    # cal.to(device)
    cal = None
    # res = None
    res = ResState()
    res.to(device)
    # res.load_state_dict(torch.load('./alldata/718nn/2022-08-02-16-16-14EKF+EKF/model.pt'))
    # cal.load_state_dict(torch.load('./alldata/718nn/2022-07-22-10-38-29smsmonecollbigcov/model.pt'))
    # params: damping x, damping y, friction x, friction y, restitution, rimfriction
    init_params = torch.tensor([0.2, 0.2, 0.01, 0.01, 0.798, 0.122], device=device)
    # init_params = cal.cal_params(torch.tensor([0., 0.], device=device))
    #  R0， R1， R2， Q01， Q23，Q4， Q5
    # covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1]).to(device=device) # original variance
    #
    covariance_params_collision = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 25e-6, 25e-2, 0.0225, 225]).to(device=device)
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 25e-8, 25e-4, 25e-6, 25e-2]).to(device=device)
    covariance_params = torch.log(covariance_params)
    covariance_params_collision = torch.log(covariance_params_collision)
    set_params = False
    set_res = True
    model = Kalman_EKF_Gradient(params=init_params, covariance_params=covariance_params, segment_size=batch_trajectory_size, device=device,
                                set_params=set_params, covariance_params_collision=covariance_params_collision)
    if set_res:
        optimizer = torch.optim.Adam(res.parameters(), lr=lr)
    elif set_params:
        optimizer = torch.optim.Adam(cal.parameters(), lr=lr)
    epoch = 0
    prepare_typ = 'EKF'
    loss_form = 'predict'
    loss_type = 'log_like'  # mse
    addition_information = '+twonet+onemodecoll'
    logdir = './alldata/83nn' + datetime.datetime.now().strftime(
        "/%Y-%m-%d-%H-%M-%S") + prepare_typ + '+' + loss_form + '+' + loss_type + addition_information
    writer = SummaryWriter(logdir)
    for t in tqdm(range(epochs)):
        # params: damping x, damping y, friction x, friction y, restitution, rimfriction
        # beta = 29 * t / epochs + 1
        beta = 15
        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=True,
                                                         type=prepare_typ, cal=cal,
                                                         beta=beta, res=res)
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []
        params = model.params
        # params = torch.abs(cal.dyna_params)
        # params = cal.cal_params(training_segment_dataset[:, 2:4]).mean(dim=0)
        writer.add_scalar('dynamics/table damping x', params[0], t)
        writer.add_scalar('dynamics/table damping y', params[1], t)
        writer.add_scalar('dynamics/table friction x', params[2], t)
        writer.add_scalar('dynamics/table friction y', params[3], t)
        writer.add_scalar('dynamics/table restitution', params[4], t)
        writer.add_scalar('dynamics/rim friction', params[5], t)
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(training_segment_dataset[index_batch], training_dataset, type=loss_form,
                                        epoch=t, cal=cal, beta=beta, res=res, loss_type=loss_type)
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
        # with torch.no_grad():
        #     plot_trajectory(abs(model.params), training_dataset, epoch=t, writer=writer, cal=cal, beta=beta, res=res,
        #                     save_weight=True)
        test_segment_dataset = model.prepare_dataset(test_dataset, type=prepare_typ, epoch=t, cal=cal, beta=beta,
                                                     res=res)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.calculate_loss(test_segment_dataset[index_batch], test_dataset, type=loss_form, epoch=t,
                                            cal=cal, beta=beta, res=res, loss_type=loss_type)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)

        if t % 2 == 0:
            if set_params:
                torch.save(cal.state_dict(), logdir + "/model.pt")
            if set_res:
                torch.save(res.state_dict(), logdir + "/model.pt")
