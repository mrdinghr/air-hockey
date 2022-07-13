import matplotlib.pyplot as plt
import numpy as np
all_trajectory = np.load('new_total_data_after_clean.npy', allow_pickle=True)
all_trajectory_part = np.load('new_total_data_after_clean_part.npy', allow_pickle=True)

def has_collision(pre, cur, next):
    if (next[0] - cur[0])*(cur[0] - pre[0]) < 0 or (next[1] - cur[1])*(cur[1] - pre[1]) < 0:
        return True
    return False


def detect_collision(trajectory):
    trajectory_batch_with_one_collision = []
    collision = False
    for i in range(len(trajectory) - 2):

        return


# for i in range(len(all_trajectory)):
#     plt.figure()
#     plt.scatter(all_trajectory[i][:, 0], all_trajectory[i][:, 1], c='b')
# for trajecotry in all_trajectory:
#     plt.figure()
#     plt.scatter(trajecotry[:, 0], trajecotry[:, 1])
# all_trajectory[2] = all_trajectory[2][50:]
# np.save('new_total_data_after_clean_part', all_trajectory)
# plt.scatter(all_trajectory[2][:, 0], all_trajectory[2][:, 1], c='b')
# plt.scatter(all_trajectory[2][50:, 0], all_trajectory[2][50:, 1])
list = [52, 237, 17, 162, 202, 152, 122, 102, 57, 67]
plt.scatter(all_trajectory_part[2][:, 0], all_trajectory_part[2][:, 1], c='r')
for i in range(len(list)):
    plt.scatter(all_trajectory_part[2][list[i]:list[i]+10, 0], all_trajectory_part[2][list[i]:list[i]+10, 1], label=str(i))
plt.legend()
plt.show()


# cut trajectory into no collision part
# trajectory_after_cut = []
# for trajectory in all_trajectory:
#     begin = 2
#     for i in range(2, len(trajectory) - 2):
#         if has_collision(trajectory[i - 1], trajectory[i], trajectory[i + 2]):
#             if i - 2 - 15 > begin:
#                 trajectory_after_cut.append(trajectory[begin:i])
#             begin = i + 2
# trajectory_after_cut = np.array(trajectory_after_cut)
# for i in range(len(trajectory_after_cut)):
#     print(len(trajectory_after_cut[i]))
# # plt.show()
# np.save('new_trajectory_after_cut', trajectory_after_cut)

