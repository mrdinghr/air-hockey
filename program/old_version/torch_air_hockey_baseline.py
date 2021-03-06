from math import pi
import torch

device = torch.device("cuda")


def cross2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


class AirHockeyTable:
    def __init__(self, length, width, goalWidth, puckRadius, restitution, rimFriction, dt):
        self.m_length = length
        self.m_width = width
        self.m_puckRadius = puckRadius
        self.m_goalWidth = goalWidth
        self.m_e = restitution
        self.m_rimFriction = rimFriction
        self.m_dt = dt

        ref = torch.tensor([length / 2, 0.])
        offsetP1 = torch.tensor([-self.m_length / 2 + self.m_puckRadius, -self.m_width / 2 + self.m_puckRadius])
        offsetP2 = torch.tensor([-self.m_length / 2 + self.m_puckRadius, self.m_width / 2 - self.m_puckRadius])
        offsetP3 = torch.tensor([self.m_length / 2 - self.m_puckRadius, -self.m_width / 2 + self.m_puckRadius])
        offsetP4 = torch.tensor([self.m_length / 2 - self.m_puckRadius, self.m_width / 2 - self.m_puckRadius])
        offsetP1 = offsetP1 + ref
        offsetP2 = offsetP2 + ref
        offsetP3 = offsetP3 + ref
        offsetP4 = offsetP4 + ref
        self.m_boundary = torch.tensor([[offsetP1[0], offsetP1[1], offsetP3[0], offsetP3[1]],
                                        [offsetP3[0], offsetP3[1], offsetP4[0], offsetP4[1]],
                                        [offsetP4[0], offsetP4[1], offsetP2[0], offsetP2[1]],
                                        [offsetP2[0], offsetP2[1], offsetP1[0], offsetP1[1]]], device=device)

        self.m_jacCollision = torch.eye(6, device=device)
        #   First Rim
        T_tmp = torch.eye(6, device=device)
        self.m_rimGlobalTransforms = torch.zeros((4, 6, 6), device=device)
        self.m_rimGlobalTransformsInv = torch.zeros((4, 6, 6), device=device)
        self.m_rimGlobalTransforms[0] = T_tmp
        self.m_rimGlobalTransformsInv[0] = torch.linalg.inv(T_tmp)
        #   Second Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][1] = -1
        T_tmp[1][0] = 1
        T_tmp[2][3] = -1
        T_tmp[3][2] = 1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[1] = T_tmp
        self.m_rimGlobalTransformsInv[1] = torch.linalg.inv(T_tmp)
        #   Third Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][0] = -1
        T_tmp[1][1] = -1
        T_tmp[2][2] = -1
        T_tmp[3][3] = -1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[2] = T_tmp
        self.m_rimGlobalTransformsInv[2] = torch.linalg.inv(T_tmp)
        #   Forth Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][1] = 1
        T_tmp[1][0] = -1
        T_tmp[2][3] = 1
        T_tmp[3][2] = -1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[3] = T_tmp
        self.m_rimGlobalTransformsInv[3] = T_tmp.T

    def set_dynamic_parameter(self, restitution, rimFriction):
        self.m_e = restitution
        self.m_rimFriction = rimFriction

    def apply_collision(self, state):
        pos = state[0:2].detach()
        vel = state[2:4].clone()
        angle = state[4].detach()
        ang_vel = state[5].clone()

        score = False
        if torch.abs(pos[1]) < self.m_goalWidth / 2 and pos[0] < self.m_boundary[0][0] + self.m_puckRadius:
            score = True
        elif torch.abs(pos[1]) < self.m_goalWidth / 2 and pos[0] > self.m_boundary[0][2] - self.m_puckRadius:
            score = True
        u = vel * self.m_dt
        cur_state = torch.zeros(6, device=device)
        for i in range(self.m_boundary.shape[0]):
            p1 = self.m_boundary[i][0:2]
            p2 = self.m_boundary[i][2:]
            v = p2 - p1  # torch.tensor([p2[0] - p1[0], p2[1] - p1[1]], device=device).double()
            w = p1 - pos  # torch.tensor([p1[0] - p[0], p1[1] - p[1]], device=device).double()
            denominator = cross2d(v, u.detach())
            if abs(denominator) < 1e-6:
                continue
            s = cross2d(v, w) / denominator
            r = cross2d(u.detach(), w) / denominator
            if cross2d(w, v) < 0 or (s >= 1e-4 and s <= 1 - 1e-4 and r >= 1e-4 and r <= 1 - 1e-4):
                state_pre = pos + s * u
                theta_pre = angle + s * ang_vel * self.m_dt
                vel.detach_()
                ang_vel.detach_()

                vecT = v / torch.linalg.norm(v)
                vecN = torch.zeros(2, device=device)
                vecN[0] = -v[1] / torch.linalg.norm(v)
                vecN[1] = v[0] / torch.linalg.norm(v)
                vtScalar = torch.dot(vel, vecT)
                vnSCalar = torch.dot(vel, vecN)
                if torch.abs(vtScalar + self.m_puckRadius * ang_vel) < 3 * self.m_rimFriction * (
                        1 + self.m_e) * torch.abs(vnSCalar):
                    # Velocity on next time step without sliding
                    vtNextSCalar = 2 * vtScalar / 3 - self.m_puckRadius * ang_vel / 3
                    vnNextScalar = -self.m_e * vnSCalar
                    # Angular volocity next point
                    cur_state5 = ang_vel / 3 - 2 * vtScalar / (3 * self.m_puckRadius)
                    # cur_state[5] = dtheta / 3 - 2 * vtScalar / (3 * self.m_puckRadius)
                    # update jacobian
                    self.m_jacCollision = torch.eye(6, device=device)
                    self.m_jacCollision[0][2] = self.m_dt
                    self.m_jacCollision[1][3] = self.m_dt
                    self.m_jacCollision[2][2] = 2 / 3
                    self.m_jacCollision[2][5] = -self.m_puckRadius / 3
                    self.m_jacCollision[3][3] = -self.m_e
                    self.m_jacCollision[4][5] = self.m_dt
                    self.m_jacCollision[5][2] = -2 / (3 * self.m_puckRadius)
                    self.m_jacCollision[5][5] = 1 / 3
                    self.m_jacCollision = self.m_jacCollision
                    jacobian = self.m_rimGlobalTransformsInv[i] @ self.m_jacCollision @ self.m_rimGlobalTransforms[i]
                    if theta_pre + (1 - s) * cur_state5 * self.m_dt > pi:
                        cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt - 2 * pi
                    elif theta_pre + (1 - s) * cur_state5 * self.m_dt < -pi:
                        cur_state[4] = 2 * pi + theta_pre + (1 - s) * cur_state5 * self.m_dt
                    else:
                        cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt
                else:
                    # velocity on next time step with sliding
                    slideDir = (vtScalar + ang_vel * self.m_puckRadius) / torch.abs(
                        vtScalar + ang_vel * self.m_puckRadius)
                    vtNextSCalar = vtScalar + self.m_rimFriction * slideDir * (1 + self.m_e) * vnSCalar
                    vnNextScalar = -self.m_e * vnSCalar
                    cur_state5 = ang_vel + 2 * self.m_rimFriction * slideDir * (
                            1 + self.m_e) * vnSCalar / self.m_puckRadius
                    # cur_state[5] = dtheta + 2 * self.m_rimFriction * slideDir * (
                    #         1 + self.m_e) * vnSCalar / self.m_puckRadius
                    self.m_jacCollision = torch.eye(6, device=device)
                    self.m_jacCollision[0][2] = self.m_dt
                    self.m_jacCollision[1][3] = self.m_dt
                    self.m_jacCollision[2][3] = self.m_rimFriction * slideDir * (1 + self.m_e)
                    self.m_jacCollision[3][3] = -self.m_e
                    self.m_jacCollision[4][5] = self.m_dt
                    self.m_jacCollision[5][3] = self.m_jacCollision[2][3] * 2 / self.m_puckRadius
                    self.m_jacCollision = self.m_jacCollision
                    jacobian = self.m_rimGlobalTransformsInv[i] @ self.m_jacCollision @ self.m_rimGlobalTransforms[i]
                    if theta_pre + (1 - s) * cur_state5 * self.m_dt > pi:
                        cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt - 2 * pi
                    elif theta_pre + (1 - s) * cur_state5 * self.m_dt < -pi:
                        cur_state[4] = 2 * pi + theta_pre + (1 - s) * cur_state5 * self.m_dt
                    else:
                        cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt

                cur_state[0:2] = state_pre + (1 - s) * (vnNextScalar * vecN + vtNextSCalar * vecT) * self.m_dt
                state.detach_()
                cur_state[2:4] = -self.m_e * vnSCalar.detach() * vecN + vtNextSCalar.detach() * vecT
                cur_state[5] = cur_state5.detach()
                # if theta + s * dtheta * self.m_dt + (1 - s) * cur_state[5] * self.m_dt > pi:
                #     cur_state[4] = theta + s * dtheta * self.m_dt + (1 - s) * cur_state[5] * self.m_dt - 2*pi
                # elif theta + s * dtheta * self.m_dt + (1 - s) * cur_state[5] * self.m_dt < -pi:
                #     cur_state[4] = 2*pi + theta + s * dtheta * self.m_dt + (1 - s) * cur_state[5] * self.m_dt
                # else:
                #     cur_state[4] = theta + s * dtheta * self.m_dt + (1 - s) * cur_state[5] * self.m_dt
                return True, cur_state, jacobian, score
        return False, cur_state, torch.eye(6, device=device), score


class SystemModel:
    def __init__(self, tableDamping, tableFriction, tableLength, tableWidth, goalWidth, puckRadius, malletRadius,
                 tableRes, malletRes, rimFriction, dt):
        self.tableDamping = tableDamping
        self.tableFriction = tableFriction
        self.tableLength = tableLength
        self.tableWidth = tableWidth
        self.goalWidth = goalWidth
        self.puckRadius = puckRadius
        self.malletRadius = malletRadius
        self.tableRes = tableRes
        self.malletRes = malletRes
        self.rimFriction = rimFriction
        self.dt = dt

    @property
    def F(self):
        J_linear = torch.eye(6, device=device)
        J_linear[0][2] = self.dt
        J_linear[1][3] = self.dt
        J_linear[2][2] = 1 - self.dt * self.tableDamping
        J_linear[3][3] = 1 - self.dt * self.tableDamping
        J_linear[4][5] = self.dt
        J_linear[5][5] = 1
        return J_linear

    def f(self, x, u):
        pos_prev = x[0:2].detach().clone()
        vel_prev = x[2:4].clone()
        ang_prev = x[4].detach().clone()
        ang_vel_prev = x[5].clone()

        pos = pos_prev + u * vel_prev
        angle = torch.fmod(ang_prev + u * ang_vel_prev, pi * 2)
        if angle > pi:
            angle = angle - pi * 2
        elif angle < -pi:
            angle = angle + pi * 2
        vel_prev.detach_()
        ang_vel_prev.detach_()
        if torch.linalg.norm(vel_prev) > 1e-6:
            vel = vel_prev - u * (self.tableDamping * vel_prev + self.tableFriction * vel_prev /
                                  torch.linalg.norm(vel_prev))
        else:
            vel = vel_prev - u * self.tableDamping * vel_prev
        ang_vel = ang_vel_prev
        return torch.cat([pos, vel, torch.atleast_1d(angle), torch.atleast_1d(ang_vel)])

    def update_jacobian(self, x, u):
        self.F = self.J_linear

    def set_damping(self, damping):
        self.tableDamping = damping

    def set_table_friction(self, mu_):
        self.tableFriction = mu_

    def set_table_dynamics_param(self, tableRes, rimFriction):
        self.tableRes = tableRes
        self.rimFriction = rimFriction

    #   self.collisionModel.setTableDynamicsParam(tableRes,rimFriction)
    def is_outside_boundary(self, measurement):
        if (abs(measurement[1]) > self.tableWidth / 2 - self.puckRadius + 0.01) or measurement[0] < -0.01 or \
                measurement[0] > self.tableLength - self.puckRadius + 0.01:
            return True
        return False
    # def is_outside_boundary(self,state):
    #     if (abs(state[1]) > self.tableWidth / 2 - self.puckRadius + 0.01) or state[0] < -0.01 or \
    #             state[0] > self.tableLength - self.puckRadius + 0.01:
    #         return True
    #     return False


'''
    def setMalletDynamicsParam(malletRestitution):
        self.malletRes=malletRestitution
        self.collisionModel.setMalletDynamics(malletRes)
'''


class Observationmodel:
    def __init__(self, state):
        self.observation = torch.tensor([state[0], state[1], state[4]])
        self.H = torch.eye(3)

    def observation(self, state):
        ob = torch.tensor([state[0], state[1], state[4]])
        return ob
