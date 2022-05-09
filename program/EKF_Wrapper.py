from filterpy.kalman import ExtendedKalmanFilter
import air_hockey_baseline
import numpy as np
table = air_hockey_baseline.AirHockeyTable(length=2, width=1, goalWidth=0.2, puckRadius=0.05, restitution=0.7,
                                           rimFriction=0.1, dt=0.01)
system = air_hockey_baseline.SystemModel(tableDamping=0.1, tableFriction=0.01, tableLength=2, tableWidth=1,
                                         goalWidth=0.2, puckRadius=0.05, malletRadius=0.1, tableRes=0.7, malletRes=0.7,
                                         rimFriction=0.1, dt=0.01)

jacobian = np.eye(6)

u = 0.01

