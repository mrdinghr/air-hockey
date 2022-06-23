from EKF_Wrapper import air_hockey_EKF
import pandas as pd
from hebo.design_space.design_space import DesignSpace
import air_hockey_baseline
import numpy as np
import matplotlib.pyplot as plt
from hebo.optimizers.bo import BO
from hebo.optimizers.hebo import HEBO
from kalman_smoothing import kalman_smooth

params = [{'name': 'tableFriction', 'type': 'num', 'lb': 0, 'ub': 0.5},
          {'name': 'tableDamping', 'type': 'num', 'lb': 0, 'ub': 0.5},
          {'name': 'tableRestitution', 'type': 'num', 'lb': 0, 'ub': 1}]
space = DesignSpace().parse(params)
bo = BO(space)
hebo = HEBO(space, rand_sample=5)


def obj(x: pd.DataFrame) -> np.ndarray:
    x = x[['tableFriction', 'tableDamping', 'tableRestitution']].values
    num_x = x.shape[0]
    ret = np.zeros((num_x, 1))
    for k in range(num_x):
        ret[k, 0] = expectation(x[k])
    return ret


def expectation(Nparams):
    system = air_hockey_baseline.SystemModel(tableDamping=Nparams[1], tableFriction=Nparams[0],
                                             tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                             puckRadius=0.03165, malletRadius=0.04815, tableRes=Nparams[2],
                                             malletRes=0.8, rimFriction=0.1418, dt=1 / 120)
    table = air_hockey_baseline.AirHockeyTable(length=1.948, width=1.038, goalWidth=0.25, puckRadius=0.03165,
                                               restitution=Nparams[2], rimFriction=0.1418, dt=1 / 120)
    raw_data = np.load("example_data.npy")
    return kalman_smooth(raw_data, system, table)


for i in range(30):
    rec_x = hebo.suggest(n_suggestions=5)
    hebo.observe(rec_x, obj(rec_x))
    min_idx = hebo.y.argmin()
    print(str(i) + ' y is ' + str(hebo.y.min()) + ' tableFriction ' + str(hebo.X.iloc[min_idx]['tableFriction']) +' tableDamping '+ str(
        hebo.X.iloc[min_idx]['tableDamping']) + ' tableRestitution ' + str(hebo.X.iloc[min_idx]['tableRestitution']))
    plt.scatter(i, hebo.y.min(), color='b')
plt.show()
