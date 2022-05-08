import matplotlib.pyplot as plt
import numpy as np

def table_plot(table):
    xy = [0, -table.m_width / 2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rect = plt.Rectangle(xy, table.m_length, table.m_width, fill=False)
    rect.set_linewidth(10)
    ax.add_patch(rect)
    plt.xlabel(
        "table width is " + str(table.m_width) + ", table length is " + str(table.m_length) + ", puck radius is " + str(
            table.m_puckRadius))


def trajectory_plot(table, system, u, x, y, dx, dy, theta, d_theta, x_var, y_var, dx_var, dy_var, theta_var, d_theta_var, state_num,
                    point_num, touchline, touch_line_x, touch_line_y):
    resx = [[]]
    resy = [[]]
    state = np.array([x, y, dx, dy, theta, d_theta])
    for j in range(point_num):
        resX = []
        resY = []
        if x_var != 0:
            state[0] = np.random.normal(x, x_var)
        elif y_var != 0:
            state[1] = np.random.normal(y, y_var)
        elif dx_var != 0:
            state[2] = np.random.normal(dx, dx_var)
        elif dy_var != 0:
            state[3] = np.random.normal(dy, dy_var)
        elif theta_var != 0:
            state[4] = np.random.normal(theta, theta_var)
        elif d_theta_var != 0:
            state[5] = np.random.normal(d_theta, d_theta_var)
        resX.append(state[0])
        resY.append(state[1])
        for i in range(state_num):
            has_collision, state =table.apply_collision(state)
            if not has_collision:
                state=system.f(state,u)
            resX.append(state[0])
            resY.append(state[1])
            if touchline:
                if state[0]*np.sign(x-touch_line_x) < touch_line_x or state[1]*np.sign(y-touch_line_y) < touch_line_y:
                    break
        resx.append(resX)
        resy.append(resY)
    for i in range(point_num):
        plt.scatter(resx[i],resy[i],alpha=0.1,c='b')
    plt.show()

