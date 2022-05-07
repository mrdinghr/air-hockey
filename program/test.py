import numpy as np
import matplotlib.pyplot as plt
import air_hockey_baseline
from tableplot import tableplot as tp

table = air_hockey_baseline.AirHockeyTable(length=2, width=1, goalWidth=0.2, puckRadius=0.1, restitution=0.7,
                                           rimFriction=0.1, dt=0.01)
tp(table)
system=air_hockey_baseline.SystemModel(tableDamping=0.1, tableFriction=0.01, tableLength=2, tableWidth=1, goalWidth=0.2, puckRadius=0.1, malletRadius=0.1,tableRes=0.7, malletRes=0.7, rimFriction=0.1, dt=0.01)
x = np.array([0.4, 0, 10, 1, 1, 0.5])  # state x y dx dy theta dtheta
jacobian = np.eye(6)
resX = []
resY = []
u = 0.01
# resX resY: record the state of x y

resX.append(x[0])
resY.append(x[1])
# has_collision, x, jacobian = table.applyCollision(x)
# print(table.applyCollision(x, jacobians))
# print(system.f(system,x,u))

for i in range(5000):
    has_collision, x = table.apply_collision(x)
    if not has_collision:
        x=system.f(x,u)
    resX.append(x[0])
    resY.append(x[1])
plt.plot(resX,resY)
plt.show()