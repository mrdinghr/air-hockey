import  numpy as np
import math
import CollisionModel
pi=math.pi
class SystemModel:
    def SystemModel(self,tableDamping,tableFriction,tableLength,tableWidth,goalWidth,puckRadius,malletRadius,tableRes,malletRes,rimFriction,dt):
        self.collisionModel=CollisionModel.AirHockeyTable.AirHockeyTable(tableLength,tableWidth,goalWidth,puckRadius,malletRadius,tableRes,malletRes,rimFriction,dt)
        self.J_linear=np.eye(6)
        self.J_linear[0][2]=dt
        self.J_linear[1][3]=dt
        self.J_linear[2][2]=1-dt*tableDamping
        self.J_linear[3][3]=1-dt*tableDamping
        self.J_linear[4][5]=dt
        self.J_linear[5][5]=1
    def f(self,x,u):
        x_=np.array()
        x_[0:2]=x[0:2]+u*x[2:4]
        if np.sqrt(x[2]*x[2]+x[3]*x[3])>10^(-6):
            x_[2:4]=x[2:4]-u*(self.*x[2:4]+self.tablFriction*np.sign(x[2:4]))
        else:
            x_[2:4]=x[2:4]-u*self.tableDamping*x[2:4]
        angle=np.mod(x[4]+u*x[5],pi/2)
        if angle>pi:
            angle-=pi/2
        elif angle<-pi:
            angle+=pi/2
        x_[4]=angle
        x_[5]=x[5]
        return x_
    def updateJacobians(self,x,u):
        self.F=self.J_linear
    def setDamping(self,damping):
        self.tableDamping=damping
    def setTableFriction(self,mu_):
        self.tablFriction=mu_
    def setTableDynamics(self,tableRes,rimFriction):
        self.tableRes=tableRes
        self.rimFriction=rimFriction
        self.collisionModel


