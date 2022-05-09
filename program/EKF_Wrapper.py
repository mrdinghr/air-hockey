from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
'''
class EKF_Wrapper:
    def __init__(self,P ,):
        self.P=P

    def predict_state(self,state,u,system,table):
        hascollision, state, jacobian=table.applycollision(state)
        if hascollision:
            system.update_jacobian(state, u)
            state=system.f(state, u)
        self.P= system.F*self.P*system.F.T
'''
def EKF_Wrapper(state, table, system, observation, measure):
    dt= system.dt
    rk=ExtendedKalmanFilter(dim_x=6, dim_z=3, dim_u=1)
    rk.x=state
    rk.F=system.F
    #rk.R=
    #rk.Q
    #rk.P
    rk.update(measure, observation.H, observation.observation)

