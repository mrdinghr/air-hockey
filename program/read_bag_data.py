import os
import rosbag
import numpy as np
import preprocess
import transformations as tr
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.dirname(__file__) + '/..')
data_dir = os.path.join(root_dir, 'rosdata')
bag = rosbag.Bag(os.path.join(data_dir, '2021-09-13-17-23-11.bag'))

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

np.save('2021-09-13-17-23-11', measurement)

fig, axes = plt.subplots(3)
axes[0].scatter(measurement[:, -1], measurement[:, 0], s=2)
axes[1].scatter(measurement[:, -1], measurement[:, 1], s=2)
axes[2].scatter(measurement[:, -1], measurement[:, 2], s=2)

fig, ax = plt.subplots(1)
ax.scatter(measurement[:, 0], measurement[:, 1], s=2)
plt.show()