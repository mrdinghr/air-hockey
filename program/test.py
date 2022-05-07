import numpy as np
import matplotlib.pyplot as plt
import air_hockey_baseline
from tableplot import tableplot as tp


table = air_hockey_baseline.AirHockeyTable(length=2, width=1, goalWidth=0.2, puckRadius=0.1, restitution=0.7,
                                           rimFriction=0.1, dt=0.01)
tp(table)
system=air_hockey_baseline.SystemModel(tableDamping=0.1, tableFriction=0.01, tableLength=2, tableWidth=1, goalWidth=0.2, puckRadius=0.1, malletRadius=0.1,tableRes=0.7, malletRes=0.7, rimFriction=0.1, dt=0.01)

jacobian = np.eye(6)

u = 0.01
resx=[[]]
resy=[[]]
'''
#code to caculate start point with gaussian distribution in x position after certain step

statesum=100 # observe after how many steps
pointsum=200 # the number of points, initialized by gaussion distribution
for j in range(pointsum):
    resX = []
    resY = []
    x = np.array([np.random.normal(0.4,0.05), 0, 10, 1, 1, 0.5])  # state x y dx dy theta dtheta
    resX.append(x[0])
    resY.append(x[1])
    for i in range(statesum):
        has_collision, x = table.apply_collision(x)
        if not has_collision:
            x=system.f(x,u)
        resX.append(x[0])
        resY.append(x[1])
    resx.append(resX[statesum])
    resy.append(resY[statesum])
plt.scatter(resx,resy,alpha=0.4)
plt.show()
'''

# code to caculate start point with gaussian distribution in x position
# observe when puck first time touch the line x=certain value
observepos=1.5  # observe after how many steps
pointsum=20  # the number of points, initialized by gaussion distribution
for j in range(pointsum):
    resX = []
    resY = []
    x = np.array([np.random.normal(0.4,0.05), 0, 2, 1, 1, 0.5])  # state x y dx dy theta dtheta
    resX.append(x[0])
    resY.append(x[1])
    while True:
        has_collision, x = table.apply_collision(x)
        if not has_collision:
            x = system.f(x, u)
        resX.append(x[0])
        resY.append(x[1])
        if x[0] >observepos:
            break
    resx.append(resX)
    resy.append(resY)
for i in range(pointsum):
    plt.scatter(resx[i], resy[i], alpha=0.1,c='b')
plt.show()

