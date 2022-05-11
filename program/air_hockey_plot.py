import matplotlib.pyplot as plt
import numpy as np


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


def trajectory_plot(table, system, u, x, y, dx, dy, theta, d_theta, x_var, y_var, dx_var, dy_var, theta_var,
                    d_theta_var, state_num,
                    puck_num, touchline, touch_line_x, touch_line_y):
    table_plot(table)
    resx = []
    resy = []
    state = np.array([x, y, dx, dy, theta, d_theta])
    for j in range(puck_num):
        resX = []
        resY = []
        state[0] = np.random.normal(x, x_var)
        state[1] = np.random.normal(y, y_var)
        state[2] = np.random.normal(dx, dx_var)
        state[3] = np.random.normal(dy, dy_var)
        state[4] = np.random.normal(theta, theta_var)
        state[5] = np.random.normal(d_theta, d_theta_var)
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
                if state[0] * np.sign(touch_line_x - x) > np.sign(touch_line_x - x) * touch_line_x or (
                        state[1] * np.sign(touch_line_y - y) > touch_line_y * np.sign(touch_line_y - y)
                        ):
                    break
        resx.append(resX)
        resy.append(resY)
    for i in range(puck_num):
        plt.scatter(resx[i], resy[i], alpha=0.1, c='b')
    return resx, resy
