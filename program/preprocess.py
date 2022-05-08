#!/usr/bin/env python
import math

import rosbag
import rospy
import transformations as tr
import numpy as np


def msg_to_se3(msg):
    se3 = tr.translation_matrix(np.array([msg.transforms[0].transform.translation.x,
                                          msg.transforms[0].transform.translation.y,
                                          msg.transforms[0].transform.translation.z])) @ \
          tr.quaternion_matrix(np.array([msg.transforms[0].transform.rotation.w,
                                         msg.transforms[0].transform.rotation.x,
                                         msg.transforms[0].transform.rotation.y,
                                         msg.transforms[0].transform.rotation.z,
                                         ]))
    return se3


def read_bag(bag):
    start, se3_world_puck_init, vel, o_vel = get_start_stamp(bag)
    start = rospy.Time(start)
    stop = get_stop_stamp(bag, start)
    measurement = []
    se3_table_world = np.eye(4)
    for _, msg, _ in bag.read_messages('/tf', start):
        if msg.transforms[0].child_frame_id == "Table":
            se3_world_table = tr.translation_matrix(np.array([msg.transforms[0].transform.translation.x,
                                                              msg.transforms[0].transform.translation.y,
                                                              msg.transforms[0].transform.translation.z])) @ \
                              tr.quaternion_matrix(np.array([msg.transforms[0].transform.rotation.w,
                                                             msg.transforms[0].transform.rotation.x,
                                                             msg.transforms[0].transform.rotation.y,
                                                             msg.transforms[0].transform.rotation.z,
                                                             ])) @ \
                              tr.translation_matrix([-0.978, 0., 0.])  # Transform from /Table(center) to /TableHome
            se3_table_world = tr.inverse_matrix(se3_world_table)
            break

    se3_table_puck_init = se3_table_world @ se3_world_puck_init
    vel_table_puck = se3_table_world[:3, :3] @ np.hstack([vel, 0.])
    start_value = np.array([se3_table_puck_init[0, 3], se3_table_puck_init[1, 3], vel_table_puck[0], vel_table_puck[1],
                            tr.euler_from_matrix(se3_table_puck_init)[-1], o_vel])

    for topic, msg, t in bag.read_messages('/tf', start, stop):
        if msg.transforms[0].child_frame_id == "Puck":
            se3_world_puck = tr.translation_matrix(np.array([msg.transforms[0].transform.translation.x,
                                                             msg.transforms[0].transform.translation.y,
                                                             msg.transforms[0].transform.translation.z])) @ \
                             tr.quaternion_matrix(np.array([msg.transforms[0].transform.rotation.w,
                                                            msg.transforms[0].transform.rotation.x,
                                                            msg.transforms[0].transform.rotation.y,
                                                            msg.transforms[0].transform.rotation.z,
                                                            ]))
            se3_table_puck = se3_table_world @ se3_world_puck
            rpy = tr.euler_from_matrix(se3_table_puck)
            measurement.append(np.array([se3_table_puck[0, 3], se3_table_puck[1, 3], rpy[-1],
                                         msg.transforms[0].header.stamp.to_sec()]))
    measurement = np.array(measurement)
    measurement[:, -1] -= measurement[0, -1]
    bag.close()
    return start_value, measurement


def normalize_angle(angle):
    if angle > math.pi:
        return angle - 2 * math.pi
    elif angle < -math.pi:
        return angle + 2 * math.pi
    else:
        return angle


def get_start_stamp(bag):
    se3_hist = []
    t_hist = []
    count = 0
    for topic, msg, t in bag.read_messages('/tf'):
        if msg.transforms[0].child_frame_id == "Puck":
            se3_world_puck = msg_to_se3(msg)
            se3_hist.append(se3_world_puck)
            t_hist.append(msg.transforms[0].header.stamp.to_sec())
            if len(se3_hist) < 15:
                continue
            else:
                vel = np.linalg.norm((se3_hist[-1][:2, 3] - se3_hist[-2][:2, 3]) / (t_hist[-1] - t_hist[-2]))
                if vel > 0.1:
                    count += 1
                    if count > 10:
                        l_vel = (se3_hist[-1][:2, 3] - se3_hist[-11][:2, 3]) / (t_hist[-1] - t_hist[-11])
                        rpy_last = tr.euler_from_matrix(se3_hist[-6])
                        rpy_begin = tr.euler_from_matrix(se3_hist[-11])
                        o_vel = normalize_angle((rpy_last[-1] - rpy_begin[-1])) / (t_hist[-6] - t_hist[-11])
                        return t_hist[-11], se3_hist[-11], l_vel, o_vel
                else:
                    count = 0
                se3_hist.pop(0)
                t_hist.pop(0)
    return False


def get_stop_stamp(bag, start):
    count = 0
    se3_hist = []
    t_hist = []

    for topic, msg, t in bag.read_messages('/tf', start):
        if msg.transforms[0].child_frame_id == "Puck":
            se3_world_puck = msg_to_se3(msg)
            se3_hist.append(se3_world_puck)
            t_hist.append(t)
            if len(se3_hist) < 15:
                continue
            else:
                vel = np.linalg.norm((se3_hist[-1][:2, 3] - se3_hist[-2][:2, 3]) / (t_hist[-1] - t_hist[-2]).to_sec())
                if vel < 1.0:
                    count += 1
                    if count > 10:
                        return t_hist[-11]
                else:
                    count = 0
                se3_hist.pop(0)
                t_hist.pop(0)
    return rospy.Time(bag.get_end_time())
