import numpy as np
import torch
from matplotlib import pyplot as plt
from air_hockey_plot import test_params_trajectory_plot
import air_hockey_baseline
import torch_air_hockey_baseline_no_detach
# from torch_EKF_Batch_gradient import calculate_init_state
from math import pi

device = torch.device("cuda")


def calculate_init_state(trajectory):
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
                          device=device).float()
    return state_


# params: damping x, damping y, friction x, friction y, restitution, rimfriction
def plot_trajectory(params, trajectories, epoch=0, writer=None, set_params=False, cal=None, beta=1, set_res=False, res=None):
    # data_set = np.load('new_total_data_after_clean.npy', allow_pickle=True)
    # data_set = np.load('example_data.npy')
    if set_params or set_res:
        params = params.cpu()
    else:
        params = params.cpu().numpy()
    for trajectory_index, data_set in enumerate(trajectories):
        # trajectory_index = index  # choose which trajectory to test, current total 150 trajectories 2022.06.21
        if set_params or set_res:
            init_state = calculate_init_state(data_set)
        else:
            init_state = calculate_init_state(data_set).cpu().numpy()
        state_num = int((data_set[-1, -1] - data_set[0, -1]) * 120)
        if set_params or set_res:
            system = torch_air_hockey_baseline_no_detach.SystemModel(tableDampingX=params[0], tableDampingY=params[1],
                                                                     tableFrictionX=params[2], tableFrictionY=params[3],
                                                                     tableLength=1.948,
                                                                     tableWidth=1.038, goalWidth=0.25,
                                                                     puckRadius=0.03165, malletRadius=0.04815,
                                                                     tableRes=params[4], malletRes=0.04815,
                                                                     rimFriction=params[5], dt=1 / 120)
            table = system.table
        else:
            table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25,
                                                       puckRadius=0.03165,
                                                       restitution=params[2], rimFriction=params[3],
                                                       dt=1 / 120)
            system = air_hockey_baseline.SystemModel(tableDamping=params[1], tableFriction=params[0],
                                                     tableLength=1.948,
                                                     tableWidth=1.038, goalWidth=0.25,
                                                     puckRadius=0.03165, malletRadius=0.04815,
                                                     tableRes=params[2], malletRes=0.04815,
                                                     rimFriction=params[3], dt=1 / 120)
        plt.figure()
        state_list, time_list = test_params_trajectory_plot(init_state=init_state, table=table, system=system,
                                                            u=1 / 120, state_num=state_num, set_params=set_params,
                                                            cal=cal, beta=beta, set_res=set_res, res=res)
        if set_params:
            state_list = torch.stack(state_list)
        else:
            state_list = np.vstack(state_list)
        # plt.figure()
        plt.scatter(data_set[:, 0], data_set[:, 1], label='record data', c='g', s=2)
        plt.scatter(data_set[0, 0], data_set[0, 1], c='r', marker='*', s=80)
        plt.legend()
        writer.add_figure('trajectory_' + str(trajectory_index) + '/prediction compare', plt.gcf(), epoch)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('x position')
        plt.scatter(data_set[:, 3] - data_set[0, 3], data_set[:, 0], label='record data', c='g', s=2)
        plt.scatter(time_list, state_list[:, 0], label='predicted data', c='b', s=2)
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.title('y position')
        plt.scatter(data_set[:, 3] - data_set[0, 3], data_set[:, 1], label='record data', c='g', s=2)
        plt.scatter(time_list, state_list[:, 1], label='predicted data', c='b', s=2)
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.title('theta')
        plt.scatter(data_set[:, 3] - data_set[0, 3], data_set[:, 2], label='record data', c='g', s=2)
        plt.scatter(time_list, state_list[:, 4], label='predicted data', c='b', s=2)
        plt.legend()
        writer.add_figure('trajectory_' + str(trajectory_index) + '/prediction ', plt.gcf(), epoch)
        plt.close()
    # plt.show()


# input: state_list:calculated by EKF and kalman smooth, correspond trajectory
# output: draw the trajectory and position, velocity
# color: r smooth b EKF g data
def plot_with_state_list(EKF_state_list, smooth_state_list, trajectory, time_list, writer=None, epoch=0,
                         trajectory_index=None):
    # EKF_state_list = torch.tensor([item.clone().cpu().numpy() for item in EKF_state_list], device=device).cpu().numpy()
    EKF_state_list = torch.stack(EKF_state_list).cpu().numpy()
    # smooth_state_list = torch.tensor([item.clone().cpu().numpy() for item in smooth_state_list], device=device).cpu().numpy()
    smooth_state_list = torch.stack(smooth_state_list).cpu().numpy()
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
            theta_velocity.append(
                (trajectory[i][2] - trajectory[i - 1][2]) / (trajectory[i][-1] - trajectory[i - 1][-1]))
    plt.figure()
    plt.scatter(trajectory[1:, 0], trajectory[1:, 1], c='g', label='recorded trajectory', alpha=0.5, s=2)
    plt.scatter(EKF_state_list[:, 0], EKF_state_list[:, 1], c='b', label='EKF trajectory', alpha=0.5, s=2)
    plt.scatter(smooth_state_list[:, 0], smooth_state_list[:, 1], c='r', label='Smooth trajectory', s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + "/cartesian", plt.gcf(), epoch)
    # position x
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x position')
    plt.scatter(time_list, EKF_state_list[:, 0], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 0], label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 0], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    # position y
    plt.subplot(3, 1, 2)
    plt.title('y position')
    plt.scatter(time_list, EKF_state_list[:, 1], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 1], label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 1], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('theta')
    plt.scatter(time_list, EKF_state_list[:, 4], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 2], label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 4], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/position', plt.gcf(), epoch)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x velocity')
    plt.scatter(time_list, EKF_state_list[:, 2], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], x_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 2], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title('y velocity')
    plt.scatter(time_list, EKF_state_list[:, 3], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], y_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 3], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('rotate velocity')
    plt.scatter(time_list, EKF_state_list[:, 5], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], theta_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 5], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/velocity', plt.gcf(), epoch)
    plt.close()
    # plt.show()


def EKF_plot_with_state_list(EKF_state_list, trajectory, writer=None, epoch=0, trajectory_index=None):
    # EKF_state_list = torch.tensor([item.clone().cpu().numpy() for item in EKF_state_list], device=device).cpu().numpy()
    EKF_state_list = torch.stack(EKF_state_list).cpu().numpy()
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
            theta_velocity.append(
                (trajectory[i][2] - trajectory[i - 1][2]) / (trajectory[i][-1] - trajectory[i - 1][-1]))
    plt.figure()
    plt.scatter(trajectory[1:, 0], trajectory[1:, 1], c='g', label='recorded trajectory', alpha=0.5)
    plt.scatter(EKF_state_list[:, 0], EKF_state_list[:, 1], c='b', label='EKF trajectory', alpha=0.5)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + "/cartesian", plt.gcf(), epoch)
    # position x
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x position')
    plt.scatter(trajectory[0:, 3] - trajectory[0, 3], EKF_state_list[:, 0], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 0], label='recorded trajectory', c='g', s=2)
    plt.legend()
    # position y
    plt.subplot(3, 1, 2)
    plt.title('y position')
    plt.scatter(trajectory[0:, 3] - trajectory[0, 3], EKF_state_list[:, 1], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 1], label='recorded trajectory', c='g', s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('theta')
    plt.scatter(trajectory[0:, 3] - trajectory[0, 3], EKF_state_list[:, 4], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 2], label='recorded trajectory', c='g', s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/position', plt.gcf(), epoch)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x velocity')
    plt.scatter(trajectory[0:, 3] - trajectory[0, 3], EKF_state_list[:, 2], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], x_velocity, label='recorded trajectory', c='g', s=2)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title('y velocity')
    plt.scatter(trajectory[0:, 3] - trajectory[0, 3], EKF_state_list[:, 3], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], y_velocity, label='recorded trajectory', c='g', s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('rotate velocity')
    plt.scatter(trajectory[0:, 3] - trajectory[0, 3], EKF_state_list[:, 5], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], theta_velocity, label='recorded trajectory', c='g', s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/velocity', plt.gcf(), epoch)
    plt.close()


if __name__ == '__main__':
    # table friction, table damping, table restitution, rim friction
    init_params = torch.tensor([0.3942, 0.394, 0.6852, 0.03275])
    index = 8
    plot_trajectory(index, init_params)
