import torch
import torch_air_hockey_baseline
from torch_EKF_Wrapper import air_hockey_EKF
from math import pi
import numpy as np
from matplotlib import pyplot as plt

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
    # plt.subplot(1, 2, 1)
    # plt.plot(result[i][:, 1])
    # plt.subplot(1, 2, 2)
    # plt.plot(result_clean[i][:, 1])
    # print(result_clean[i].shape, result[i].shape)
    # plt.show()
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
