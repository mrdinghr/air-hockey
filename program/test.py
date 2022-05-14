import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import air_hockey_baseline
from air_hockey_plot import trajectory_plot as trajectory
from air_hockey_plot import table_plot

table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                           restitution=0.7424, rimFriction=0.1418, dt=1 / 120)
system = air_hockey_baseline.SystemModel(tableDamping=0.001, tableFriction=0.001, tableLength=1.948, tableWidth=1.038,
                                         goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
                                         tableRes=0.7424, malletRes=0.8, rimFriction=0.1418, dt=1 / 120)

jacobian = np.eye(6)

u = 1/120
resx, resy = trajectory(table=table, system=system, u=u, x=1.38578465, y=-0.31278382, dx=(1.41637898-1.38578465)/(3.59138727-3.58327246), dy=(0.31278382-0.28991479)/(3.59138727-3.58327246), theta=2.94323937, d_theta=(3.59138727-3.58327246)/(3.59138727-3.58327246), x_var=0.0,
                        y_var=0.0,
                        dx_var=0., dy_var=0., theta_var=0, d_theta_var=0, state_num=10000,
                        puck_num=1, touchline=False, touch_line_x=2, touch_line_y=2)
plt.show()
x_final = []
y_final = []
for i in resx:
    x_final.append(i[-1])
for i in resy:
    y_final.append(i[-1])
# print("mean of x " + str(np.mean(resx)) + " mean of y " + str(np.mean(resy)) + " var of x " + str(
#     np.var(resx)) + " var of y " + str(np.var(resy)))
ax = plt.subplot(111)
# table_plot(table)  # plot the table square
cov = np.cov(x_final, y_final)
eigen_value, v = np.linalg.eig(cov)
eigen_value = np.sqrt(eigen_value)
s = 9.21  # confidence interval 99 9.21  95  5.991  90 4.605
ell = matplotlib.patches.Ellipse(xy=(np.mean(x_final), np.mean(y_final)), width=eigen_value[0] * np.sqrt(s) * 2,
                                 height=eigen_value[1] * np.sqrt(s) * 2, angle=np.rad2deg(np.arccos(v[0, 0])),
                                 alpha=0.3)
ax.add_artist(ell)
plt.xlim((0, table.m_length))
plt.ylim((-table.m_width / 2, table.m_width / 2))
plt.scatter(x_final, y_final)
plt.axis('scaled')
plt.axis('equal')
# plt.xlabel("var of final x " + str(np.var(resx)) + " var of final y " + str(np.var(resy)))
plt.title("x y as gaussian variable x_var,y_var=0.02 stop after 100 step")
plt.show()
