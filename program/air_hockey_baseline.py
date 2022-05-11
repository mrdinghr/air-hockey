import numpy as np
import math
import numpy.linalg as lg

pi = math.pi


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

        ref = np.array([length / 2, 0.])
        offsetP1 = np.array([-self.m_length / 2 + self.m_puckRadius, -self.m_width / 2 + self.m_puckRadius])
        offsetP2 = np.array([-self.m_length / 2 + self.m_puckRadius, self.m_width / 2 - self.m_puckRadius])
        offsetP3 = np.array([self.m_length / 2 - self.m_puckRadius, -self.m_width / 2 + self.m_puckRadius])
        offsetP4 = np.array([self.m_length / 2 - self.m_puckRadius, self.m_width / 2 - self.m_puckRadius])
        offsetP1 += ref
        offsetP2 += ref
        offsetP3 += ref
        offsetP4 += ref
        self.m_boundary = np.array([[offsetP1[0], offsetP1[1], offsetP3[0], offsetP3[1]],
                                    [offsetP3[0], offsetP3[1], offsetP4[0], offsetP4[1]],
                                    [offsetP4[0], offsetP4[1], offsetP2[0], offsetP2[1]],
                                    [offsetP2[0], offsetP2[1], offsetP1[0], offsetP1[1]]])

        collisionRim = -1
        self.m_jacCollision = np.eye(6)
        #   First Rim
        T_tmp = np.eye(6)
        self.m_rimGlobalTransforms = np.zeros((4, 6, 6))
        self.m_rimGlobalTransformsInv = np.zeros((4, 6, 6))
        self.m_rimGlobalTransforms[0] = T_tmp
        self.m_rimGlobalTransformsInv[0] = lg.inv(T_tmp)
        #   Second Rim
        T_tmp = np.zeros((6, 6))
        T_tmp[0][1] = -1
        T_tmp[1][0] = 1
        T_tmp[2][3] = -1
        T_tmp[3][2] = 1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[1] = T_tmp
        self.m_rimGlobalTransformsInv[1] = lg.inv(T_tmp)
        #   Third Rim
        T_tmp = np.zeros((6, 6))
        T_tmp[0][0] = -1
        T_tmp[1][1] = -1
        T_tmp[2][2] = -1
        T_tmp[3][3] = -1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[2] = T_tmp
        self.m_rimGlobalTransformsInv[2] = lg.inv(T_tmp)
        #   Forth Rim
        T_tmp = np.zeros((6, 6))
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
        p = state[0:2]
        vel = state[2:4]
        jacobian = np.eye(6)
        if abs(p[1]) < self.m_goalWidth / 2 and p[0] < self.m_boundary[0][0] + self.m_puckRadius:
            return False, state, jacobian, True
        elif abs(p[1]) < self.m_goalWidth / 2 and p[0] > self.m_boundary[0][2] - self.m_puckRadius:
            return False, state, jacobian, True
        u = vel * self.m_dt
        i = 0
        for i in range(self.m_boundary.shape[0]):
            p1 = self.m_boundary[i][0:2]
            p2 = self.m_boundary[i][2:]
            v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            w = np.array([p1[0] - p[0], p1[1] - p[1]])
            denominator = cross2d(v, u)
            if abs(denominator) < 1e-6:
                continue
            s = cross2d(v, w) / denominator
            r = cross2d(u, w) / denominator
            if cross2d(w, v) < 0 or (
                    s >= 1e-4 and s <= 1 - 1e-4 and r >= 1e-4 and r <= 1 - 1e-4):
                theta = state[4]
                dtheta = state[5]
                collisionRim = i
                vecT = v / np.sqrt(v[0] * v[0] + v[1] * v[1])
                vecN = np.zeros(2)
                vecN[0] = -v[1] / np.sqrt(v[0] * v[0] + v[1] * v[1])
                vecN[1] = v[0] / np.sqrt(v[0] * v[0] + v[1] * v[1])
                vtScalar = np.dot(vel, vecT)
                vnSCalar = np.dot(vel, vecN)
                if abs(vtScalar + self.m_puckRadius * dtheta) < 3 * self.m_rimFriction * (1 + self.m_e) * abs(vnSCalar):
                    # Velocity on next time step without sliding
                    vtNextSCalar = 2 * vtScalar / 3 - self.m_puckRadius * dtheta / 3
                    vnNextScalar = -self.m_e * vnSCalar
                    # Angular volocity next point
                    state[5] = dtheta / 3 - 2 * vtScalar / (3 * self.m_puckRadius)
                    # update jacobian
                    self.m_jacCollision = np.eye(6)
                    self.m_jacCollision[0][2] = self.m_dt
                    self.m_jacCollision[1][3] = self.m_dt
                    self.m_jacCollision[2][2] = 2 / 3
                    self.m_jacCollision[2][5] = -self.m_puckRadius / 3
                    self.m_jacCollision[3][3] = -self.m_e
                    self.m_jacCollision[4][5] = self.m_dt
                    self.m_jacCollision[5][2] = -2 / (3 * self.m_puckRadius)
                    self.m_jacCollision[5][5] = 1 / 3
                    jacobian = self.m_rimGlobalTransformsInv[i] @ self.m_jacCollision @ self.m_rimGlobalTransforms[i]
                else:
                    # velocity on next time step with sliding
                    slideDir = (vtScalar + dtheta * self.m_puckRadius) / abs(vtScalar + dtheta * self.m_puckRadius)
                    vtNextSCalar = vtScalar + self.m_rimFriction * slideDir * (1 + self.m_e) * vnSCalar
                    vnNextScalar = -self.m_e * vnSCalar
                    state[5] = dtheta + 2 * self.m_rimFriction * slideDir * (
                            1 + self.m_e) * vnSCalar / self.m_puckRadius
                    self.m_jacCollision = np.eye(6)
                    self.m_jacCollision[0][2] = self.m_dt
                    self.m_jacCollision[1][3] = self.m_dt
                    self.m_jacCollision[2][3] = self.m_rimFriction * slideDir * (1 + self.m_e)
                    self.m_jacCollision[3][3] = -self.m_e
                    self.m_jacCollision[4][5] = self.m_dt
                    self.m_jacCollision[5][3] = -2 / (3 * self.m_puckRadius)
                    jacobian = self.m_rimGlobalTransformsInv[i] @ self.m_jacCollision @ self.m_rimGlobalTransforms[i]
                state[2:4] = vnNextScalar * vecN + vtNextSCalar * vecT
                state[0:2] = p + s * u + (1 - s) * state[2:4] * self.m_dt
                state[4] = theta + s * dtheta * self.m_dt + (1 - s) * state[5] * self.m_dt
                return True, state, jacobian, False
        return False, state, jacobian, False


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
        # self.collisionModel=
        self.J_linear = np.eye(6)
        self.J_linear[0][2] = dt
        self.J_linear[1][3] = dt
        self.J_linear[2][2] = 1 - dt * tableDamping
        self.J_linear[3][3] = 1 - dt * tableDamping
        self.J_linear[4][5] = dt
        self.J_linear[5][5] = 1
        self.F = self.J_linear

    def f(self, x, u):
        x_ = np.zeros(6)
        x_[0:2] = x[0:2] + u * x[2:4]
        if np.sqrt(x[2] * x[2] + x[3] * x[3]) > 1e-6:
            x_[2:4] = x[2:4] - u * (self.tableDamping * x[2:4] + self.tableFriction * np.sign(x[2:4]))
        else:
            x_[2:4] = x[2:4] - u * self.tableDamping * x[2:4]
        angle = np.mod(x[4] + u * x[5], pi / 2)
        if angle > pi:
            angle -= pi / 2
        elif angle < -pi:
            angle += pi / 2
        x_[4] = angle
        x_[5] = x[5]
        return x_

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
        self.observation = np.array([state[0], state[1], state[4]])
        self.H = np.eye(3)

    def observation(self, state):
        ob = np.array([state[0], state[1], state[4]])
        return ob
