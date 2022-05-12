import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import air_hockey_baseline
from air_hockey_plot import trajectory_plot as trajectory
from air_hockey_plot import table_plot

table = air_hockey_baseline.AirHockeyTable(length=2, width=1, goalWidth=0.1, puckRadius=0.05, restitution=0.7,
                                           rimFriction=0.5, dt=0.01)
system = air_hockey_baseline.SystemModel(tableDamping=0., tableFriction=0.4, tableLength=2, tableWidth=1,
                                         goalWidth=0.2, puckRadius=0.05, malletRadius=0.1, tableRes=0.7, malletRes=0.7,
                                         rimFriction=0.5, dt=0.01)

jacobian = np.eye(6)

u = 0.01
resx, resy = trajectory(table=table, system=system, u=u, x=0.5, y=0, dx=10, dy=4, theta=2, d_theta=0., x_var=0.02,
                        y_var=0.02,
                        dx_var=0., dy_var=0., theta_var=0, d_theta_var=0, state_num=1000,
                        puck_num=10, touchline=False, touch_line_x=2, touch_line_y=2)
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
# plt.axis('equal')
# plt.xlabel("var of final x " + str(np.var(resx)) + " var of final y " + str(np.var(resy)))
plt.title("x y as gaussian variable x_var,y_var=0.02 stop after 100 step")
plt.show()
