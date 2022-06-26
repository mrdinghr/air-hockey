import numpy as np
from matplotlib import pyplot as plt
from air_hockey_plot import test_params_trajectory_plot
import air_hockey_baseline
from torch_EKF_Batch_gradient import calculate_init_state
data_set = np.load('total_data_after_clean.npy', allow_pickle=True)
# data_set = np.load('example_data.npy')
params = np.array([5.2457e-3, 0.4198, 0.644, 0.05251])  # table friction, table damping, table restitution, rim friction
trajectory_index = 69  # choose which trajectory to test, current total 150 trajectories 2022.06.21
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
