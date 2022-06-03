from math import pi
import torch
device = torch.device("cuda")


class air_hockey_EKF:
    def __init__(self, state, u, system, table, Q, R, P):
        self.state = state
        self.system = system
        self.table = table
        self.Q = Q
        self.Q.to(device)
        self.R = R
        self.R.to(device)
        self.P = P
        self.P.to(device)
        self.u = u
        self.predict_state = None
        self.F = None
        self.score = False
        self.has_collision = False
        self.H = torch.zeros((3, 6), device=device)
        self.H[0][0] = self.H[1][1] = self.H[2][4] = 1

    def predict(self):
        self.P = self.system.F @ self.P @ self.system.F.T + self.Q
        self.P.to(device)
        self.has_collision, self.predict_state, jacobian, self.score = self.table.apply_collision(self.state)
        if self.has_collision:
            self.F = jacobian
        else:
            self.F = self.system.F
            self.predict_state = (self.system.f(self.state, self.u)).float()

    def update(self, measure):
        # measurement residual
        H = torch.zeros((3, 6), device=device)
        H[0][0] = H[1][1] = H[2][4] = 1
        self.y = (measure - torch.tensor([self.predict_state[0], self.predict_state[1], self.predict_state[4]], device=device)).float()
        if abs(self.y[2]) > pi:
            self.y[2] = self.y[2] - torch.sign(measure[2]) * 2 * pi
        self.S = H @ self.P @ H.T + self.R
        K = (self.P @ H.T @ torch.inverse(self.S)).float()
        self.state = (self.predict_state + K @ self.y).float()
        self.P = (torch.eye(6, device=device) - K @ H) @ self.P