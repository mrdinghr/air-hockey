import os
import rosbag
import numpy as np
import preprocess
import transformations as tr
import matplotlib.pyplot as plt


cmp = np.load('2021-09-13-17-23-11.npy')
root_dir = os.path.abspath(os.path.dirname(__file__) + '/..')
data_dir = os.path.join(root_dir, 'puck_record')
file_name = os.listdir(data_dir)
result = []
for cur_file_name in file_name:
    bag = rosbag.Bag(os.path.join(data_dir, cur_file_name))
    se3_table_world = None
    measurement = []
    for topic, msg, t in bag.read_messages('/tf'):
        if se3_table_world is None and msg.transforms[0].child_frame_id == "Table":
            se3_world_table = preprocess.msg_to_se3(msg)
            se3_table_world = tr.inverse_matrix(se3_world_table)

        if se3_table_world is not None and msg.transforms[0].child_frame_id == "Puck":
            se3_world_puck = preprocess.msg_to_se3(msg)
            se3_table_puck = se3_table_world @ se3_world_puck
            rpy = tr.euler_from_matrix(se3_table_puck)
            measurement.append(np.array([se3_table_puck[0, 3], se3_table_puck[1, 3], rpy[-1],
                                         msg.transforms[0].header.stamp.to_sec()]))

    measurement = np.array(measurement)
    measurement[:, -1] -= measurement[0, -1]
    result.append(measurement)
np.save('new_total_data', result)

'''
#  clean all trajectory data: throw no move part
table_length = 1.948
result = np.load('new_total_data.npy', allow_pickle=True)
result_clean = [[] for i in range(len(result))]
for i in range(len(result)):

    for j in range(1, len(result[i])):
        if abs(result[i][j][0] - result[i][j - 1][0]) < 0.005 and abs(result[i][j][1] - result[i][j - 1][1]) < 0.005:
            continue
        if result_clean[i] != []:
            if result[i][j][3] - result_clean[i][-1][3] > 5 / 120:
                break
        result_clean[i].append(result[i][j])
result_clean = np.array(result_clean)
for i in range(len(result_clean)):
    for i_data in result_clean[i]:
        i_data[0] += table_length / 2
for i in range(len(result_clean)):
    result_clean[i] = np.array(result_clean[i])
for i in range(len(result_clean)):
    plt.figure()
    plt.scatter(result_clean[i][:, 3], result_clean[i][:, 0], c='b')
plt.show()
np.save('new_total_data_after_clean', result_clean)
'''
