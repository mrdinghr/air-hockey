from math import pi

import numpy as np
import torch

device = torch.device("cuda")


def cross2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


class AirHockeyTable:
    def __init__(self, length, width, goalWidth, puckRadius, restitution, rimFriction, dt, tableDampingX,
                 tableDampingY):
        self.m_length = length
        self.m_width = width
        self.m_puckRadius = puckRadius
        self.m_goalWidth = goalWidth
        self.m_e = restitution
        self.m_rimFriction = rimFriction
        self.m_dt = dt
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY

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
        T_tmp[0][1] = 1
        T_tmp[1][0] = -1
        T_tmp[2][3] = 1
        T_tmp[3][2] = -1
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
        T_tmp[0][1] = -1
        T_tmp[1][0] = 1
        T_tmp[2][3] = -1
        T_tmp[3][2] = 1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[3] = T_tmp
        self.m_rimGlobalTransformsInv[3] = T_tmp.T

    def set_dynamic_parameter(self, restitution, rimFriction, tableDampingX, tableDampingY):
        self.m_e = restitution
        self.m_rimFriction = rimFriction
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY

    def collision_outof_boundary(self, i, pos, vel):
        if i == 0:
            if pos[1] <= self.m_boundary[i][1] and vel[1] < 0:
                return True
            else:
                return False
        if i == 1:
            if pos[0] >= self.m_boundary[i][0] and vel[0] > 0:
                return True
            else:
                return False
        if i == 2:
            if pos[1] >= self.m_boundary[i][1] and vel[1] > 0:
                return True
            else:
                return False
        if i == 3:
            if pos[0] <= self.m_boundary[i][0] and vel[0] < 0:
                return True
            else:
                return False

    def collision_in_boundary(self, s, r, pos):
        if ((self.m_boundary[2][:2] - pos > 0).all() and (self.m_boundary[0][:2] - pos < 0).all() and (
                s >= 1e-4 and s <= 1 - 1e-4 and r >= 1e-4 and r <= 1 - 1e-4)):
            return True
        else:
            return False

    def apply_collision(self, state, beta=1, save_weight=False, writer=None, epoch=0, collision_time=0):
        pos = state[0:2]
        vel = state[2:4]
        angle = state[4]
        ang_vel = state[5]
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
            # denominator = cross2d(v, u.detach())
            denominator = cross2d(v, u)
            if abs(denominator) < 1e-6:
                continue
            s = cross2d(v, w) / denominator
            # r = cross2d(u.detach(), w) / denominator
            r = cross2d(u, w) / denominator
            if self.collision_in_boundary(s, r, pos) or self.collision_outof_boundary(i, pos, vel):
                if self.collision_outof_boundary(i, pos, vel):
                    s = 0
                state_pre = pos + s * u
                theta_pre = angle + s * ang_vel * self.m_dt
                vecT = v / torch.linalg.norm(v)
                vecN = torch.stack([-v[1] / torch.linalg.norm(v), v[0] / torch.linalg.norm(v)]).to(device=device)
                vtScalar = torch.dot(vel, vecT)
                vnSCalar = torch.dot(vel, vecN)

                # if torch.abs(vtScalar + self.m_puckRadius * ang_vel) < 3 * self.m_rimFriction * (
                #         1 + self.m_e) * torch.abs(vnSCalar):
                #     # Velocity on next time step without sliding
                #     vtNextSCalar = 2 * vtScalar / 3 - self.m_puckRadius * ang_vel / 3
                #     vnNextScalar = -self.m_e * vnSCalar
                #     # Angular volocity next point
                #     cur_state5 = ang_vel / 3 - 2 * vtScalar / (3 * self.m_puckRadius)
                #     # cur_state[5] = dtheta / 3 - 2 * vtScalar / (3 * self.m_puckRadius)
                #     # update jacobian
                #     self.m_jacCollision = torch.eye(6, device=device)
                #     self.m_jacCollision[0][2] = self.m_dt
                #     self.m_jacCollision[1][3] = self.m_dt
                #     self.m_jacCollision[2][2] = 2 / 3
                #     self.m_jacCollision[2][5] = -self.m_puckRadius / 3
                #     self.m_jacCollision[3][3] = -self.m_e
                #     self.m_jacCollision[4][5] = self.m_dt
                #     self.m_jacCollision[5][2] = -2 / (3 * self.m_puckRadius)
                #     self.m_jacCollision[5][5] = 1 / 3
                #     jacobian = self.m_rimGlobalTransformsInv[i] @ self.m_jacCollision @ self.m_rimGlobalTransforms[i]
                #     if theta_pre + (1 - s) * cur_state5 * self.m_dt > pi:
                #         cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt - 2 * pi
                #     elif theta_pre + (1 - s) * cur_state5 * self.m_dt < -pi:
                #         cur_state[4] = 2 * pi + theta_pre + (1 - s) * cur_state5 * self.m_dt
                #     else:
                #         cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt
                # else:
                #     # velocity on next time step with sliding
                #     slideDir = torch.sign(vtScalar + ang_vel * self.m_puckRadius)
                #     vtNextSCalar = vtScalar + self.m_rimFriction * slideDir * (1 + self.m_e) * vnSCalar
                #     vnNextScalar = -self.m_e * vnSCalar
                #     cur_state5 = ang_vel + 2 * self.m_rimFriction * slideDir * (
                #             1 + self.m_e) * vnSCalar / self.m_puckRadius
                #     # cur_state[5] = dtheta + 2 * self.m_rimFriction * slideDir * (
                #     #         1 + self.m_e) * vnSCalar / self.m_puckRadius
                #     self.m_jacCollision = torch.eye(6, device=device)
                #     self.m_jacCollision[0][2] = self.m_dt
                #     self.m_jacCollision[1][3] = self.m_dt
                #     self.m_jacCollision[2][3] = self.m_rimFriction * slideDir * (1 + self.m_e)
                #     self.m_jacCollision[3][3] = -self.m_e
                #     self.m_jacCollision[4][5] = self.m_dt
                #     self.m_jacCollision[5][3] = self.m_jacCollision[2][3] * 2 / self.m_puckRadius
                #     jacobian = self.m_rimGlobalTransformsInv[i] @ self.m_jacCollision @ self.m_rimGlobalTransforms[i]
                #     if theta_pre + (1 - s) * cur_state5 * self.m_dt > pi:
                #         cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt - 2 * pi
                #     elif theta_pre + (1 - s) * cur_state5 * self.m_dt < -pi:
                #         cur_state[4] = 2 * pi + theta_pre + (1 - s) * cur_state5 * self.m_dt
                #     else:
                #         cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt
                # weight = 3 * self.m_rimFriction * (1 + self.m_e) * torch.abs(vnSCalar) - torch.abs(
                #     vtScalar + self.m_puckRadius * ang_vel)
                # weight = torch.sigmoid(beta * weight)
                weight = 0
                if save_weight:
                    writer.add_scalar('weight/weight' + str(collision_time), weight, epoch)
                    collision_time += 1
                slideDir = torch.sign(vtScalar + ang_vel * self.m_puckRadius)
                vtNextSCalar = weight * (2 * vtScalar / 3 - self.m_puckRadius * ang_vel / 3) + (1 - weight) * (
                        vtScalar + self.m_rimFriction * slideDir * (1 + self.m_e) * vnSCalar)
                vnNextScalar = -self.m_e * vnSCalar
                cur_state5 = weight * (ang_vel / 3 - 2 * vtScalar / (3 * self.m_puckRadius)) + (1 - weight) * (
                        ang_vel + 2 * self.m_rimFriction * slideDir * (1 + self.m_e) * vnSCalar / self.m_puckRadius)
                # if 3 * self.m_rimFriction * (1 + self.m_e) * torch.abs(vnSCalar) - torch.abs(
                #     vtScalar + self.m_puckRadius * ang_vel) > 0:
                m_jacCollision_mode_no_slide = torch.eye(6, device=device)
                m_jacCollision_mode_no_slide[2][2] = 2 / 3
                m_jacCollision_mode_no_slide[2][5] = -self.m_puckRadius / 3
                m_jacCollision_mode_no_slide[3][3] = -self.m_e
                m_jacCollision_mode_no_slide[5][2] = -2 / (3 * self.m_puckRadius)
                m_jacCollision_mode_no_slide[5][5] = 1 / 3
                # m_jacCollision = m_jacCollision_mode_no_slide
                # else:
                mu = self.m_rimFriction
                # mu = min(self.m_rimFriction,
                #           abs(vtScalar + self.m_puckRadius * ang_vel) / (3 * (1 + self.m_e) * torch.abs(vnSCalar)))
                m_jacCollision_mode_slide = torch.eye(6, device=device)
                m_jacCollision_mode_slide[2][3] = mu * slideDir * (1 + self.m_e)
                m_jacCollision_mode_slide[3][3] = -self.m_e
                m_jacCollision_mode_slide[5][3] = mu * slideDir * (1 + self.m_e) * 2 / self.m_puckRadius
                # m_jacCollision = m_jacCollision_mode_slide
                m_jacCollision = weight * m_jacCollision_mode_no_slide + (1 - weight) * m_jacCollision_mode_slide
                jacobian_global = self.m_rimGlobalTransformsInv[i] @ m_jacCollision @ self.m_rimGlobalTransforms[i]

                F_pre_collision = torch.eye(6, device=device)
                F_pre_collision[0][2] = s * self.m_dt
                F_pre_collision[1][3] = s * self.m_dt
                F_pre_collision[2][2] = 1 - s * self.m_dt * self.tableDampingX
                F_pre_collision[3][3] = 1 - s * self.m_dt * self.tableDampingY
                F_pre_collision[4][5] = s * self.m_dt
                F_pre_collision[5][5] = 1
                F_post_collision = torch.eye(6, device=device)
                F_post_collision[0][2] = (1 - s) * self.m_dt
                F_post_collision[1][3] = (1 - s) * self.m_dt
                F_post_collision[2][2] = 1 - (1 - s) * self.m_dt * self.tableDampingX
                F_post_collision[3][3] = 1 - (1 - s) * self.m_dt * self.tableDampingY
                F_post_collision[4][5] = (1 - s) * self.m_dt
                F_post_collision[5][5] = 1
                jacobian = F_post_collision @ jacobian_global @ F_pre_collision
                if jacobian[4] @ state > pi:
                    cur_state[4] = jacobian[4] @ state - 2 * pi
                    # cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt - 2 * pi
                elif jacobian[4] @ state < -pi:
                    # cur_state[4] = 2 * pi + theta_pre + (1 - s) * cur_state5 * self.m_dt
                    cur_state[4] = jacobian[4] @ state + 2 * pi
                else:
                    # cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt
                    cur_state[4] = jacobian[4] @ state
                cur_state[0:2] = jacobian[0:2] @ state
                cur_state[2:4] = jacobian[2:4] @ state
                cur_state[5] = jacobian[5] @ state
                if save_weight:
                    return True, cur_state, jacobian, score, collision_time
                return True, cur_state, jacobian, score
        if save_weight:
            return False, state, torch.eye(6, device=device), score, collision_time
        return False, state, torch.eye(6, device=device), score


class SystemModel:
    def __init__(self, tableDampingX, tableDampingY, tableFrictionX, tableFrictionY, tableLength, tableWidth, goalWidth,
                 puckRadius, malletRadius,
                 tableRes, malletRes, rimFriction, dt):
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY
        self.tableFrictionX = tableFrictionX
        self.tableFrictionY = tableFrictionY
        self.tableLength = tableLength
        self.tableWidth = tableWidth
        self.goalWidth = goalWidth
        self.puckRadius = puckRadius
        self.malletRadius = malletRadius
        self.dt = dt
        self.table = AirHockeyTable(length=tableLength, width=tableWidth, goalWidth=goalWidth,
                                    puckRadius=puckRadius, restitution=tableRes, rimFriction=rimFriction, dt=dt,
                                    tableDampingX=tableDampingX, tableDampingY=tableDampingY)

    @property
    def F(self):
        J_linear = torch.eye(6, device=device)
        J_linear[0][2] = self.dt
        J_linear[1][3] = self.dt
        J_linear[2][2] = 1 - self.dt * self.tableDampingX
        J_linear[3][3] = 1 - self.dt * self.tableDampingY
        J_linear[4][5] = self.dt
        J_linear[5][5] = 1
        return J_linear

    def f(self, x, u):
        pos_prev = x[0:2]
        vel_prev = x[2:4]
        ang_prev = x[4]
        ang_vel_prev = x[5]

        pos = pos_prev + u * vel_prev
        angle = torch.fmod(ang_prev + u * ang_vel_prev, pi * 2)
        if angle > pi:
            angle = angle - pi * 2
        elif angle < -pi:
            angle = angle + pi * 2
        # vel = vel_prev - u * self.tableDamping * vel_prev
        if torch.linalg.norm(vel_prev) > 1e-6:
            vel = vel_prev - u * (torch.stack([self.tableDampingX, self.tableDampingY]) * vel_prev +
                                  torch.stack([self.tableFrictionX, self.tableFrictionY]) * torch.sign(vel_prev))
        else:
            vel = vel_prev - u * torch.stack([self.tableDampingX, self.tableDampingY]) * vel_prev
        # ang_vel = ang_vel_prev - ang_vel_prev * self.puckRadius ** 2 * self.tableDamping * u / 4
        ang_vel = ang_vel_prev
        return torch.cat([pos, vel, torch.atleast_1d(angle), torch.atleast_1d(ang_vel)])

    def set_params(self, tableDampingX, tableDampingY, tableFrictionX, tableFrictionY, restitution, rimFriction):
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY
        self.tableFrictionX = tableFrictionX
        self.tableFrictionY = tableFrictionY
        self.table.set_dynamic_parameter(tableDampingX=tableDampingX, tableDampingY=tableDampingY,
                                         rimFriction=rimFriction, restitution=restitution)

    def apply_collision(self, state, beta=1):
        return self.table.apply_collision(state, beta=beta)
