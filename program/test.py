import numpy as np
from matplotlib import pyplot as plt
from air_hockey_plot import test_params_trajectory_plot
import air_hockey_baseline
from torch_EKF_Batch_gradient import calculate_init_state
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from test_params import plot_trajectory
from torch_gradient import load_dataset
device = torch.device("cuda")

# all_trajectory = np.load('new_total_data_after_clean.npy')
file_name = 'new_total_data_after_clean_part.npy'
training_dataset, test_dataset = load_dataset(file_name)
plot_trajectory(torch.tensor([0.2, 0.2, 0.01, 0.01, 0.798, 0.122], device=device), training_dataset)
# def has_collision(pre, cur, next):
#     if


# writer = SummaryWriter('./alldata/77test')
# print(writer != None)

# table friction, table damping, table restitution, rim friction
# init_params = torch.Tensor([0.10033657, 0.15026346, 0.79968596, 0.10029725])
# index = 5
# plot_trajectory(index, init_params)


# data = np.load('new_total_data_after_clean.npy', allow_pickle=True)
# for trajectory in data:
#     plt.figure()
#     plt.scatter(trajectory[:, 3], trajectory[:, 0], c='b')
# plt.show()

'''
# some no used code
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
'''

# data = np.load('new_total_data_after_clean.npy', allow_pickle=True)
# for i in range(len(data)):
#     plt.figure()
#     plt.plot(data[i][:, 0], data[i][:,1])
# plt.show()
# a = torch.tensor([0.,1.,2.])
# a.requires_grad= True
# b = torch.zeros(2)
# b[1] = a[1]
# b[0] = a[2]
# b = torch.zeros(2)
# b[1] = a[0]
# b[0] = a[0]
#
# print(b)




# t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# a = []

# class double_dataset(Dataset):
#     def __init__(self, dataset1, dataset2):
#         self.dataset1 = dataset1
#         self.dataset2 = dataset2
#
#     def __getitem__(self, index):
#         x1 = self.dataset1[index]
#         x2 = self.dataset2[index]
#         return x1, x2
#
#     def __len__(self):
#         return len(self.dataset1)


# dataset = double_dataset(t, d)
# loader = Data.DataLoader(dataset, batch_size=3)
# for batch in loader:
#     print(batch)
# b = a.clone().detach()


"""
#  not completely write code about torch kalman smooth
def state_kalman_smooth(cur_trajectory, dyna_params, covariance_params):
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
    data = cur_trajectory
    u = 1/120
    puck_EKF = air_hockey_EKF(u, system, table, Q, R, P)
    EKF_res_state = []
    EKF_res_P = []
    EKF_res_dynamic = []
    EKF_res_collision = []
    EKF_res_update = []
    i = 0
    j = 1
    length = len(data)
    time_EKF = []
    while j < length:
        i += 1
        time_EKF.append(i / 120)
        puck_EKF.predict()


    return
"""

# writer = SummaryWriter('./bgd')
# for i in range(10):
#     writer.add_scalar('test', i, i)
#     writer.close()


# data_set = np.load('total_data_after_clean.npy', allow_pickle=True)
# params = np.array([0.25, 0.3, 0.6, 0.8])  # table friction, table damping, table restitution, rim friction
# trajectory_index = 50  # choose which trajectory to test, current total 150 trajectories 2022.06.21
# init_state = calculate_init_state(data_set[trajectory_index]).cpu().numpy()
# state_num = 1000
# table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
#                                            restitution=params[2], rimFriction=params[3], dt=1 / 120)
# system = air_hockey_baseline.SystemModel(tableDamping=params[1], tableFriction=params[0], tableLength=1.948,
#                                          tableWidth=1.038, goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
#                                          tableRes=params[2], malletRes=0.04815, rimFriction=params[3], dt=1 / 120)
# test_params_trajectory_plot(init_state=init_state, table=table, system=system, u=1/120, state_num=state_num)
# plt.scatter(data_set[trajectory_index][:, 0], data_set[trajectory_index][:, 1], label='record data', c='r')
# plt.legend()
# plt.show()

# torch.set_printoptions(precision=8)
# device = torch.device("cuda")
# y = torch.tensor([0, 3, 6, 9, 12])
# x = torch.tensor([0, 1, 2, 3, 4])
# k = torch.tensor([2.], requires_grad=True)
#
#
# class Klinear(torch.nn.Module):
#     def __init__(self, para):
#         super(Klinear, self).__init__()
#         self.register_parameter('k', torch.nn.Parameter(para))
#         self.a = para
#
#     def make_loss_list(self, y, x):
#         loss_list = []
#         for i in range(10):
#             x.detach_()
#             x = y - self.k * x
#             loss_list.append(torch.mean(x))
#         return loss_list
#
#
# model = Klinear(k)
# lr = 0.1
# opt = torch.optim.Adam(model.parameters(), lr=lr)
# for t in range(3):
#     print(t)
#     loss_list = model.make_loss_list(y, x)
#     # dataset = Data.IterableDataset(np.arange(len(loss_list)))
#     loader = Data.DataLoader(loss_list, batch_size=2, shuffle=False)
#     for loss_batch in loader:
#         opt.zero_grad()
#         sum_loss = torch.mean(loss_batch)
#         sum_loss.backward(retain_graph=True)
#         print(model.get_parameter('k'))
#         print(model.get_parameter('k').grad)
#         opt.step()
# import torch
# import torch_air_hockey_baseline
# from torch_EKF_Wrapper import air_hockey_EKF
# from math import pi
# import numpy as np
# from matplotlib import pyplot as plt
# import torch.utils.data as Data
#
# torch.set_printoptions(precision=8)
# device = torch.device("cuda")
# y = torch.tensor([0, 3, 6, 9, 12])
# x = torch.tensor([0, 1, 2, 3, 4])
# k = torch.tensor([2.], requires_grad=True)
#
#
# class Klinear(torch.nn.Module):
#     def __init__(self, para):
#         super(Klinear, self).__init__()
#         self.register_parameter('k', torch.nn.Parameter(para))
#         self.a = para
#     def make_loss_list(self, y, x):
#         a = self.get_parameter('k')*self.get_parameter('k')
#         loss = y - a * x
#         return loss
#
#
# model = Klinear(k)
# lr = 0.1
# opt = torch.optim.Adam(model.parameters(), lr=lr)
# for t in range(3):
#     print(t)
#     loss_list = model.make_loss_list(y, x)
#     dataset = Data.TensorDataset(loss_list)
#     loader = Data.DataLoader(dataset=dataset, batch_size=2, shuffle=False)
#     for loss_batch in loader:
#         opt.zero_grad()
#         sum_loss = torch.mean(loss_batch[0])
#         sum_loss.backward(retain_graph=True)
#         print(model.get_parameter('k'))
#         print(model.get_parameter('k').grad)
#         opt.step()





# a = result[0]
# plt.plot(result[20][:, 0], result[20][:, 1])
# plt.show()

# table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
#                                            restitution=0.7424, rimFriction=0.1418, dt=1 / 120)
# system = air_hockey_baseline.SystemModel(tableDamping=0.001, tableFriction=0.001, tableLength=1.948, tableWidth=1.038,
#                                          goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
#                                          tableRes=0.7424, malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
#
# jacobian = np.eye(6)
#
# u = 1 / 120
# state_num = 100
# puck_num = 100
# x_var = 0.0
# y_var = 0.0
# dx_var = 0.0
# dy_var = 0.0
# theta_var = 0
# d_theta_var = 15
# resx, resy = trajectory(table=table, system=system, u=u, x=0.4, y=0, dx=2, dy=-2, theta=0, d_theta=50, x_var=x_var,
#                         y_var=y_var,
#                         dx_var=dx_var, dy_var=dy_var, theta_var=theta_var, d_theta_var=d_theta_var, state_num=state_num,
#                         puck_num=puck_num, touchline=False, touch_line_x=2, touch_line_y=2)
# plt.show()
# x_final = []
# y_final = []
# for i in resx:
#     x_final.append(i[-1])
# for i in resy:
#     y_final.append(i[-1])
# # print("mean of x " + str(np.mean(resx)) + " mean of y " + str(np.mean(resy)) + " var of x " + str(
# #     np.var(resx)) + " var of y " + str(np.var(resy)))
# ax = plt.subplot(111)
# cov = np.cov(x_final, y_final)
# eigen_value, v = np.linalg.eig(cov)
# eigen_value = np.sqrt(eigen_value)
# s = 9.21  # confidence interval 99 9.21  95  5.991  90 4.605
# ell = matplotlib.patches.Ellipse(xy=(np.mean(x_final), np.mean(y_final)), width=eigen_value[0] * np.sqrt(s) * 2,
#                                  height=eigen_value[1] * np.sqrt(s) * 2, angle=np.rad2deg(np.arccos(v[0, 0])),
#                                  alpha=0.3)
# ax.add_artist(ell)
# # plot the table square
# xy = [0, -table.m_width / 2]
# rect = plt.Rectangle(xy, table.m_length, table.m_width, fill=False)
# rect.set_linewidth(10)
# ax.add_patch(rect)
# # plot the table square
# ax.scatter(x_final, y_final)
# plt.xlim((0, table.m_length))
# plt.ylim((-table.m_width / 2, table.m_width / 2))
# plt.axis('scaled')
# plt.axis('equal')
# plt.xlabel("var of final x " + str(np.var(x_final)) + " var of final y " + str(np.var(y_final)))
# plt.title(
#     str(puck_num) + " pucks with dtheta as gaussian variable; dtheta var is " + str(d_theta_var) + " stop after " + str(
#         state_num) + " step")
# plt.show()
