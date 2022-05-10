import numpy.linalg as lg

class air_hockey_EKF:
    def __init__(self, state, u, system, table, observation, Q, R, P):
        self.state = state
        self.system = system
        self.table = table
        self.observation = observation
        self.Q = Q
        self.R = R
        self.P = P
        self.u = u
        self.predict_state=None

    def predict(self, state, u):
        self.P = self.system.F @ self.P @ self.system.F.T + self.Q
        self.predict_state = self.system.f(self.state, self.u)

    def update(self,z):
        #measurement residual
        y=z-self.observation(self.state)
        S=self.P+self.R #innovation covariance  because H is identify matrix so ignored
        K=self.P@lg.inv(S)
        self.state=self.predict_state+K@y
        self.P=self.F@self.P@self.F.T+self.Q





