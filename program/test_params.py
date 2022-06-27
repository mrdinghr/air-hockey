import numpy as np
import torch
from matplotlib import pyplot as plt
from air_hockey_plot import test_params_trajectory_plot
import air_hockey_baseline
from torch_EKF_Batch_gradient import calculate_init_state
from math import pi
device = torch.device("cuda")


# table friction, table damping, table restitution, rim friction
def plot_trajectory(index, params):
    data_set = np.load('total_data_after_clean.npy', allow_pickle=True)
    # data_set = np.load('example_data.npy')
    params = params.cpu().numpy()
    trajectory_index = index  # choose which trajectory to test, current total 150 trajectories 2022.06.21
    init_state = calculate_init_state(data_set[trajectory_index]).cpu().numpy()
    state_num = 1000
    table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                               restitution=params[2], rimFriction=params[3], dt=1 / 120)
    system = air_hockey_baseline.SystemModel(tableDamping=params[1], tableFriction=params[0], tableLength=1.948,
                                             tableWidth=1.038, goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
                                             tableRes=params[2], malletRes=0.04815, rimFriction=params[3], dt=1 / 120)
    test_params_trajectory_plot(init_state=init_state, table=table, system=system, u=1/120, state_num=state_num)
    plt.scatter(data_set[trajectory_index][:, 0], data_set[trajectory_index][:, 1], label='record data', c='r')
    plt.scatter(data_set[trajectory_index][0, 0], data_set[trajectory_index][0, 1], c='g', marker='*', s=80)
    plt.legend()
    plt.show()


# input: state_list:calculated by EKF and kalman smooth, correspond trajectory
# output: draw the trajectory and position, velocity
# color: r smooth b EKF g data
def plot_with_state_list(EKF_state_list, smooth_state_list, trajectory, time_list):
    EKF_state_list = torch.tensor([item.cpu().numpy() for item in EKF_state_list], device=device).cpu().numpy()
    smooth_state_list = torch.tensor([item.cpu().numpy() for item in smooth_state_list], device=device).cpu().numpy()
    trajectory = trajectory.cpu().numpy()
    x_velocity = []
    y_velocity = []
    theta_velocity = []
    for i in range(1, len(trajectory)):
        x_velocity.append((trajectory[i][0] - trajectory[i - 1][0]) / (trajectory[i][3] - trajectory[i - 1][3]))
        y_velocity.append((trajectory[i][1] - trajectory[i - 1][1]) / (trajectory[i][3] - trajectory[i - 1][3]))
        if abs(trajectory[i][2] - trajectory[i - 1][2]) > pi:
            theta_velocity.append(
                (trajectory[i][2] - np.sign(trajectory[i][2]) * pi) / (trajectory[i][-1] - trajectory[i - 1][-1]))
        else:
            theta_velocity.append((trajectory[i][2] - trajectory[i - 1][2]) / (trajectory[i][-1] - trajectory[i - 1][-1]))
    plt.figure()
    plt.scatter(trajectory[1:, 0], trajectory[1:, 1], c='g', label='recorded trajectory', alpha=0.5)
    plt.scatter(EKF_state_list[:, 0], EKF_state_list[:, 1], c='b', label='EKF trajectory', alpha=0.5)
    plt.scatter(smooth_state_list[:, 0], smooth_state_list[:, 1], c='r', label='Smooth trajectory')
    plt.legend()
    # position x
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x position')
    plt.scatter(time_list, EKF_state_list[:, 0], label='EKF trajectory', c='b')
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 0], label='recorded trajectory', c='g')
    plt.scatter(trajectory[2:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 0], label='Smooth trajectory', c='r')
    plt.legend()
    # position y
    plt.subplot(3, 1, 2)
    plt.title('y position')
    plt.scatter(time_list, EKF_state_list[:, 1], label='EKF trajectory', c='b')
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 1], label='recorded trajectory', c='g')
    plt.scatter(trajectory[2:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 1], label='Smooth trajectory', c='r')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('theta')
    plt.scatter(time_list, EKF_state_list[:, 4], label='EKF trajectory', c='b')
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 2], label='recorded trajectory', c='g')
    plt.scatter(trajectory[2:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 4], label='Smooth trajectory', c='r')
    plt.legend()
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x velocity')
    plt.scatter(time_list, EKF_state_list[:, 2], label='EKF trajectory', c='b')
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], x_velocity, label='recorded trajectory', c='g')
    plt.scatter(trajectory[2:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 2], label='Smooth trajectory', c='r')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title('y velocity')
    plt.scatter(time_list, EKF_state_list[:, 3], label='EKF trajectory', c='b')
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], y_velocity, label='recorded trajectory', c='g')
    plt.scatter(trajectory[2:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 3], label='Smooth trajectory', c='r')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('rotate velocity')
    plt.scatter(time_list, EKF_state_list[:, 5], label='EKF trajectory', c='b')
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], theta_velocity, label='recorded trajectory', c='g')
    plt.scatter(trajectory[2:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 5], label='Smooth trajectory', c='r')
    plt.legend()
    plt.show()


