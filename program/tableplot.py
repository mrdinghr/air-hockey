import matplotlib.pyplot as plt
def tableplot(table):
    xy=[0,-table.m_width/2]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    rect=plt.Rectangle(xy,table.m_length,table.m_width,fill=False)
    ax.add_patch(rect)
    plt.xlabel("table width is "+str(table.m_width)+", table length is "+str(table.m_length))



