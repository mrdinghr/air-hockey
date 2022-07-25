import matplotlib.pyplot as plt
import numpy as np
import torch


# need to set title and plt.show() after using this function
# table_plot: draw the desktop frame
# trajectory_plot: draw the trajectory of amount of puck_num pucks.
#               puck state is initialized by x y dx dy theta d_theta
#               x_var, y_var, dx_var, dy_var, theta_var, d_theta_var is to decide which variable is gaussian variable
#               touchline: criteria to stop. when True, stop when touchn line x=touch_line_x or y=touch_line_y
#               when False, stop after state_num step

def table_plot(table):
    xy = [0, -table.m_width / 2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rect = plt.Rectangle(xy, table.m_length, table.m_width, fill=False)
    rect.set_linewidth(10)
    ax.add_patch(rect)
    plt.ylabel(
        "table width is " + str(table.m_width) + ", table length is " + str(table.m_length) + ", puck radius is " + str(
            table.m_puckRadius))


def trajectory_plot(table, system, u, state, x_var, y_var, dx_var, dy_var, theta_var,
                    d_theta_var, state_num,
                    puck_num, touchline, touch_line_x, touch_line_y):
    table_plot(table)
    resx = []
    resy = []
    for j in range(puck_num):
        resX = []
        resY = []
        state[0] = np.random.normal(state[0], x_var)
        state[1] = np.random.normal(state[1], y_var)
        state[2] = np.random.normal(state[2], dx_var)
        state[3] = np.random.normal(state[3], dy_var)
        state[4] = np.random.normal(state[4], theta_var)
        state[5] = np.random.normal(state[5], d_theta_var)
        resX.append(state[0])
        resY.append(state[1])
        for i in range(state_num):
            has_collision, state, jacobian, score = table.apply_collision(state)
            if score:
                break
            if not has_collision:
                state = system.f(state, u)
            resX.append(state[0])
            resY.append(state[1])
            if touchline:
                if state[0] * np.sign(touch_line_x - state[0]) > np.sign(touch_line_x - state[0]) * touch_line_x or (
                        state[1] * np.sign(touch_line_y - state[1]) > touch_line_y * np.sign(touch_line_y - state[1])
                ):
                    break
        resx.append(resX)
        resy.append(resY)
    for i in range(puck_num):
        plt.scatter(resx[i], resy[i], alpha=0.1, c='b')
    return resx, resy


def test_params_trajectory_plot(init_state, table, system, u, state_num, set_params=False, cal=None, beta=1,
                                set_res=False, res=None):
    table_plot(table)
    resX = []
    resY = []
    if set_params:
        res_state = [init_state.cpu()]
    else:
        res_state = [init_state]
    state = init_state
    time_list = [1 / 120]
    for i in range(state_num):
        if set_params:
            params = cal.cal_params(torch.stack([state[0], state[1]]))
            system.set_params(tableDampingX=params[0], tableDampingY=params[1], tableFrictionX=params[2],
                              tableFrictionY=params[3], restitution=params[4],
                              rimFriction=params[5])
            has_collision, predict_state, jacobian, score = table.apply_collision(state, beta=beta)
        elif set_res:
            has_collision, predict_state, jacobian, score = table.apply_collision(state, beta=beta)
        else:
            has_collision, predict_state, jacobian, score = table.apply_collision(state)
        # if score:
        #     break
        if not has_collision:
            predict_state = system.f(state, u)
        if set_res:
            predict_state = res.cal_res(state) + predict_state
        if set_params:
            resX.append(predict_state[0].cpu())
            resY.append(predict_state[1].cpu())
            res_state.append(predict_state.cpu())
        else:
            resX.append(predict_state[0])
            resY.append(predict_state[1])
            res_state.append(predict_state)
        time_list.append((i + 2) / 120)
        state = predict_state
    plt.scatter(resX, resY, alpha=0.1, c='b', label='predict state by params', s=2)
    return res_state, time_list
