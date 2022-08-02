import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rosbag
from scipy.signal import filtfilt, butter
from scipy.fft import rfft, irfft, rfftfreq


def read_bag(bag, duration):
    time = []
    position = []
    effort = []
    start = False
    count = 0
    t_start = -1

    for topic, msg, t_bag in bag.read_messages():
        if not start:
            if topic in ["/iiwa_front/joint_position_trajectory_controller/state",
                         "/iiwa_front/joint_feedforward_trajectory_controller/state"]:
                if np.linalg.norm(msg.desired.velocities) > 1e-4:
                    start = True
                else:
                    continue
        elif topic in ["/iiwa_front/joint_states", ]:
            if t_start < 0:
                t_start = msg.header.stamp.to_sec()
            if count < duration * 1000:
                count += 1
                position.append(msg.position)
                effort.append(msg.effort)
                t_cur = msg.header.stamp.to_sec() - t_start
                time.append([t_cur])
            else:
                break
    return np.array(time), np.array(position), np.array(effort)


def get_vel_acc(t, q_m, tau_m, visualize=True, plot_joint_id=0):
    dq_m = np.zeros_like(q_m)
    ddq_m = np.zeros_like(q_m)

    # FFT Filtering
    sample_frequency = 1000
    q_s = rfft(q_m, axis=0)
    freq = rfftfreq(q_m.shape[0], 1/sample_frequency)
    cut_off_freq = 2
    idx = np.where(freq > cut_off_freq)
    q_s[idx] = 0
    q_fft_filter = irfft(q_s, axis=0)

    dq_s = q_s.copy()
    ddq_s = q_s.copy()
    for k, q_k in enumerate(q_s[np.where(freq <= cut_off_freq)]):
        omega_k = 2 * np.pi * (k) * sample_frequency / q_m.shape[0]
        dq_s[k] = q_k * omega_k * (1.j)
        ddq_s[k] = -q_k * omega_k ** 2

    dq_fft_filter = irfft(dq_s, axis=0)
    ddq_fft_filter = irfft(ddq_s, axis=0)

    # Position
    b, a = butter(4, 5, fs=1000)
    q_m_filter = filtfilt(b, a, q_m, axis=0)

    # Velocity
    dq_m[1:-1] = (q_m[2:, :] - q_m[:-2, :]) / (t[2:] - t[:-2])
    dq_m[0] = dq_m[1]
    dq_m[-1] = dq_m[-2]
    dq_m_filter = filtfilt(b, a, dq_m, axis=0)

    # Acceleration
    ddq_m[1:-1] = (dq_m_filter[2:, :] - dq_m_filter[:-2, :]) / (t[2:] - t[:-2])
    ddq_m[0] = ddq_m[1]
    ddq_m[-1] = ddq_m[-2]
    ddq_m_filter = filtfilt(b, a, ddq_m, axis=0)

    # # Torque
    # tau_m = -tau_m
    # b, a = butter(4, 15, fs=1000)
    # tau_m_filter = filtfilt(b, a, tau_m, axis=0)
    tau_m_filter = tau_m  # Don't need to filter the torque as the noise is gaussian distributed

    if visualize:
        fig, axes = plt.subplots(2)
        fig.suptitle("Position")
        axes[0].plot(t, q_m[:, plot_joint_id], label='original')
        axes[0].plot(t, q_m_filter[:, plot_joint_id], label='filtered')
        axes[0].plot(t, q_fft_filter[:, plot_joint_id], label='frequency filtered')
        axes[1].hist(q_m_filter[:, plot_joint_id] - q_m[:, plot_joint_id], bins=100, label='error')
        axes[0].legend()
        axes[1].legend()

        fig, axes = plt.subplots(2)
        fig.suptitle("Velocity")
        axes[0].plot(t, dq_m[:, plot_joint_id], label='original_fd')
        axes[0].plot(t, dq_m_filter[:, plot_joint_id], label='origin_fd_filtered')
        axes[0].plot(t, dq_fft_filter[:, plot_joint_id], label='frequency filtered')
        axes[1].hist(dq_m_filter[:, plot_joint_id] - dq_m[:, plot_joint_id], bins=200, label='origin_fd_error',
                     alpha=0.5)
        axes[0].legend()
        axes[1].legend()

        fig, axes = plt.subplots(2)
        fig.suptitle("Acceleration")
        axes[0].plot(t, ddq_m[:, plot_joint_id], label='filter_fd')
        axes[0].plot(t, ddq_m_filter[:, plot_joint_id], label='filter_fd_filter')
        axes[0].plot(t, ddq_fft_filter[:, plot_joint_id], label='frequency filtered')
        axes[1].hist(ddq_m_filter[:, plot_joint_id] - ddq_m[:, plot_joint_id], bins=200, label='filter_fd_error',
                     alpha=0.5)
        axes[0].legend()
        axes[1].legend()

        fig, axes = plt.subplots(2)
        fig.suptitle("Torque")
        axes[0].scatter(t, tau_m[:, plot_joint_id], label='filter_fd', s=0.5)
        axes[0].plot(t, tau_m_filter[:, plot_joint_id], label='filter_fd_filter', c='tab:orange')
        axes[1].hist(tau_m_filter[:, plot_joint_id] - tau_m[:, plot_joint_id], bins=200, label='filter_fd_error',
                     alpha=0.5)
        axes[0].legend()
        axes[1].legend()

        fig, axes = plt.subplots(4, sharex=True)
        fig.suptitle("FFT")

        freq = np.fft.fftfreq(ddq_m.shape[0], d=0.001)
        n = 300

        q_fft = np.fft.fft(q_m[:, plot_joint_id])
        dq_fft = np.fft.fft(dq_m[:, plot_joint_id])
        ddq_fft = np.fft.fft(ddq_m[:, plot_joint_id])
        tau_fft = np.fft.fft(tau_m[:, plot_joint_id])

        axes[0].bar(freq[:n], np.abs(q_fft[:n].real), width=0.1)
        axes[1].bar(freq[:n], np.abs(dq_fft[:n].real), width=0.1)
        axes[2].bar(freq[:n], np.abs(ddq_fft[:n].real), width=0.1)
        axes[3].bar(freq[:n], np.abs(tau_fft[:n].real), width=0.1)

        plt.show()

    # return q_m_filter, dq_m_filter, ddq_m_filter, tau_m_filter
    return q_fft_filter, dq_fft_filter, ddq_fft_filter, tau_m_filter


def filter_data(visualize=False, plot_joint_id=1, duration=10.2):
    traj_dir = os.path.abspath("../excitation_trajectory/record/" + traj)
    for file in os.listdir(traj_dir):
        if file.endswith(".bag"):
            bag = rosbag.Bag(os.path.join(traj_dir, file))
            t, q_m, tau_m = read_bag(bag, duration)  # Measure the position and torque
            q_f, dq_f, ddq_f, tau_f = get_vel_acc(t, q_m, tau_m, visualize, plot_joint_id=plot_joint_id)

            column_name = ['time',
                           'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7',
                           'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6', 'dq7',
                           'ddq1', 'ddq2', 'ddq3', 'ddq4', 'ddq5', 'ddq6', 'ddq7',
                           't1', 't2', 't3', 't4', 't5', 't6', 't7']
            df = pd.DataFrame(np.hstack([t, q_f, dq_f, ddq_f, tau_f]), columns=column_name)

            save_path = os.path.join(traj_dir.replace(traj, ""), "fft_filter/", traj)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df.to_csv(os.path.join(save_path, file.replace('.bag', '.csv')), index=False)


def plot_filtered_trajectories():
    traj_dir = os.path.abspath("../excitation_trajectory/record/fft_filter/" + traj)

    fig, axes = plt.subplots(4, 7, sharex=True, figsize=(50, 25))
    for file in os.listdir(traj_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(traj_dir, file))
            for joint_id in range(7):
                axes[0, joint_id].set_title("joint_" + str(joint_id + 1), fontdict={'size': 30})
                axes[0, joint_id].plot(df.time, df['q' + str(joint_id + 1)], lw=0.05)
                axes[1, joint_id].plot(df.time, df['dq' + str(joint_id + 1)], lw=0.05)
                axes[2, joint_id].plot(df.time, df['ddq' + str(joint_id + 1)], lw=0.05)
                axes[3, joint_id].plot(df.time, df['t' + str(joint_id + 1)], alpha=0.5, lw=0.05)
    axes[0, 0].set_ylabel('positions', fontdict={'size': 30})
    axes[1, 0].set_ylabel('velocities', fontdict={'size': 30})
    axes[2, 0].set_ylabel('accelerations', fontdict={'size': 30})
    axes[3, 0].set_ylabel('torque', fontdict={'size': 30})
    axes[3, 3].set_xlabel('t', fontdict={'size': 30})
    plt.savefig(os.path.join(traj_dir, "trajectory.pdf"), bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    traj = "traj_24"
    filter_data(visualize=False, plot_joint_id=6, duration=10.)
    plot_filtered_trajectories()
