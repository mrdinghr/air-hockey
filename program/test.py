import matplotlib.pyplot as plt
import numpy as np
import air_hockey_baseline
from air_hockey_plot import trajectory_plot as trajectory

table = air_hockey_baseline.AirHockeyTable(length=2, width=1, goalWidth=0.2, puckRadius=0.05, restitution=0.7,
                                           rimFriction=0.1, dt=0.01)
system = air_hockey_baseline.SystemModel(tableDamping=0.1, tableFriction=0.01, tableLength=2, tableWidth=1,
                                         goalWidth=0.2, puckRadius=0.05, malletRadius=0.1, tableRes=0.7, malletRes=0.7,
                                         rimFriction=0.1, dt=0.01)

jacobian = np.eye(6)

u = 0.01
trajectory(table=table, system=system, u=u, x=0.5, y=0, dx=2, dy=2, theta=2, d_theta=1, x_var=0.05, y_var=0.05,
           dx_var=0, dy_var=0, theta_var=0, d_theta_var=0, state_num=200,
           puck_num=10, touchline=True, touch_line_x=2, touch_line_y=-0.1)
plt.title("x y as gaussian variable")
plt.show()
