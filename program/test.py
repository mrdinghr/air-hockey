import numpy as np
from matplotlib import pyplot as plt
from air_hockey_plot import test_params_trajectory_plot
import air_hockey_baseline
from torch_EKF_Batch_gradient import calculate_init_state
from torch.utils.tensorboard import SummaryWriter
import torch
a = torch.tensor([1,2, 3,4])
b=c=d=a
e=[a,b]
f =[c,d]
plt.scatter(e[:][0], f[:][0])
# b = a.clone().detach()

plt.show()


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

'''
#  clean all trajectory data: throw no move part
table_length = 1.948
result = np.load('total_data.npy', allow_pickle=True)
result_clean = [[] for i in range(len(result))]
for i in range(len(result)):
    for j in range(1, len(result[i])):
        if abs(result[i][j][0] - result[i][j - 1][0]) < 0.005 and abs(result[i][j][1] - result[i][j - 1][1]) < 0.005:
            continue
        result_clean[i].append(result[i][j])
result_clean = np.array(result_clean)
for i in range(len(result_clean)):
    for i_data in result_clean[i]:
        i_data[0] += table_length / 2
for i in range(len(result_clean)):
    result_clean[i] = np.array(result_clean[i])
np.save('total_data_after_clean', result_clean)
'''

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
