import matplotlib.pyplot as plt
import numpy as np
all_trajectory = np.load('new_total_data_after_clean.npy', allow_pickle=True)


def has_collision(pre, cur, next):
    if (next[0] - cur[0])*(cur[0] - pre[0]) < 0 or (next[1] - cur[1])*(cur[1] - pre[1]) < 0:
        return True
    return False


trajectory_after_cut = []
for trajectory in all_trajectory:
    begin = 2
    for i in range(2, len(trajectory) - 2):
        if has_collision(trajectory[i - 1], trajectory[i], trajectory[i + 2]):
            if i - 2 - 5 > begin:
                trajectory_after_cut.append(trajectory[begin:i])
            begin = i + 2
trajectory_after_cut = np.array(trajectory_after_cut)
for i in range(len(trajectory_after_cut)):
    plt.figure()
    plt.scatter(trajectory_after_cut[i][:, 0], trajectory_after_cut[i][:, 1])
plt.show()
np.save('new_trajectory_after_cut', trajectory_after_cut)

