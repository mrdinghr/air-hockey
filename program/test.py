import numpy as np
import matplotlib.pyplot as plt

import air_hockey_baseline

table = air_hockey_baseline.AirHockeyTable(length=2, width=1, goalWidth=0.2, puckRadius=0.1, restitution=0.7,
                                           rimFriction=0.1, dt=0.01)
system=air_hockey_baseline.SystemModel(tableDamping=0.1, tableFriction=0.01, tableLength=2, tableWidth=1, goalWidth=0.2, puckRadius=0.1, malletRadius=0.1,tableRes=0.7, malletRes=0.7, rimFriction=0.1, dt=0.01)
# here set length=1 width=0.5 goalwidth                    tableRes, malletRes, rimFriction, dt=0.2 pucketradius=0.1 restitution=0.5 rimfrcition=0.1 dt=0.05
x = np.array([0.5, 0, 3, 2, 1, 0.5])  # state x y dx dy theta dtheta
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
    has_collision, x = table.applyCollision(x)
    if not has_collision:
        x=system.f(x,u)
    resX.append(x[0])
    resY.append(x[1])
print(resX)
print(len(resY))
print(resY)
plt.plot(resX,resY)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
