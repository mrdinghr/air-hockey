import numpy as np
import numpy.linalg as lg
class AirHockeyTable:
       def cross2d(v1, v2):
        return v1[0]*v2[1]-v1[1]*v2[0]
       def AirHockeyTable(self,length,width,goalWidth,puckRadius,restitution,rimFriction,dt):

            self.m_length=length
            self.m_width=width
            self.m_puckRadius=puckRadius
            self.m_goalWidth=goalWidth
            self.m_e=restitution
            self.m_rimFriction=rimFriction
            self.m_dt=dt
            ref=np.array([length/2,length/2])
            offsetP1=np.array([-self.m_length/2+self.m_puckRadius,-self.m_width/2+self.m_puckRadius])
            offsetP2=np.array([-self.m_length/2+self.m_puckRadius,self.m_width/2-self.m_puckRadius])
            offsetP3 = np.array([self.m_length/2-self.m_puckRadius,-self.m_width/2+self.m_puckRadius])
            offsetP4 = np.array([self.m_length / 2 -self.m_puckRadius, self.m_width / 2 - self.m_puckRadius])
            offsetP1+=ref
            offsetP2+=ref
            offsetP3+=ref
            offsetP4+=ref
            self.m_boundary=np.array([[offsetP1[0],offsetP1[1],offsetP3[0],offsetP3[1]],
                                 [offsetP3[0],offsetP3[1],offsetP4[0],offsetP4[1]],
                                 [offsetP4[0],offsetP4[1],offsetP2[0],offsetP2[1]],
                                 offsetP2[0],offsetP2[1],offsetP1[0],offsetP1[1]])
            collisionRim=-1
            self.m_jacCollision=np.eye(6)
        #   First Rim
            T_tmp=np.eye(6)
            self.m_rimGlobalTransforms=np.zeros((4,6,6))
            self.m_rimGlobalTransformsInv=np.zeros((4,6,6))
            self.m_rimGlobalTransforms[0]=T_tmp
            self.m_rimGlobalTransformsInv[0]=lg.inv(T_tmp)
        #   Second Rim
            T_tmp=np.zeros((6,6))
            T_tmp[0][1]=-1
            T_tmp[1][0]=1
            T_tmp[2][3]=-1
            T_tmp[3][2]=1
            T_tmp[4][4]=1
            T_tmp[5][5]=1
            self.m_rimGlobalTransforms[1]=T_tmp
            self.m_rimGlobalTransformsInv[1]=lg.inv(T_tmp)
        #   Third Rim
            T_tmp=np.zeros((6,6))
            T_tmp[0][0]=-1
            T_tmp[1][1]=-1
            T_tmp[2][2]=-1
            T_tmp[3][3]=-1
            T_tmp[4][4]=1
            T_tmp[5][5]=1
            self.m_rimGlobalTransforms[2] = T_tmp
            self.m_rimGlobalTransformsInv[2] = lg.inv(T_tmp)
        #   Forth Rim
            T_tmp = np.zeros((6, 6))
            T_tmp[0][1]=1
            T_tmp[1][0]=-1
            T_tmp[2][3]=1
            T_tmp[3][2]=-1
            T_tmp[4][4]=1
            T_tmp[5][5]=1
            self.m_rimGlobalTransforms[3] = T_tmp
            self.m_rimGlobalTransformsInv[3] = lg.inv(T_tmp)
       def setDynamichsParameter(self,restitution,rimFriction):
            self.m_e=restitution
            self.m_rimFriction=rimFriction
#     state 6*1 x,y,dx,dy,theta,dtheta
       def applyCollision(self,state,jacobian):
            p=state[0:2]
            vel=state[2:4]
            if (p[1])<self.m_goalWidth/2 and p[0]<self.m_boundary[0][0]+self.m_puckRadius:
                return False
            elif abs(p[1])<self.m_goalWidth/2 and p[0]>self.m_boundary[0][2]-self.m_puckRadius:
                return False
            u=vel*self.m_dt
            i=0
            while i<4:
                 p1=self.m_boundary[i][0:2]
                 p2=self.m_boundary[i][2:]
                 v=p2-p1
                 w=p1-p
                 denominator=self.cross2d(v,u)
                 s=self.cross2d(v,w)/denominator
                 r=self.cross2d(u,w)/denominator
                 if abs(denominator)<1e-6:
                      continue
                 if self.cross2d(w,v)<0 or (s>=10^(-4) and s<=1-10^(-4) and r>=10^(-4) and r<=1-10^(-4)):
                      theta=state[4]
                      dtheta=state[5]
                      collisionRim=i
                      vecT=v/np.sqrt(v[0]*v[0]+v[1]*v[1])
                      vecN=np.array()
                      vecN[0]=-v[1]/np.sqrt(v[0]*v[0]+v[1]*v[1])
                      vecN[1]=v[0]/np.sqrt(v[0]*v[0]+v[1]*v[1])
                      vtScalar=np.dot(vel,vecT)
                      vnSCalar=np.dot(vel,vecN)
                      if abs(vtScalar+self.m_puckRadius*dtheta)<3*self.m_rimFriction*(1+self.m_e)*abs(vnSCalar):
                           #Velocity on next time step without sliding
                           vtNextSCalar=2*vtScalar/3-self.m_puckRadius*dtheta/3
                           vnNextScalar=-self.m_e*vnSCalar
                           #Angular volocity next point
                           state[5]=dtheta/3-2*vtScalar/(3*self.m_puckRadius)
                           # update jacobian
                           self.m_jacCollision=np.eye(6)
                           self.m_jacCollision[0][2]=self.m_dt
                           self.m_jacCollision[1][3]=self.m_dt
                           self.m_jacCollision[2][2]=2/3
                           self.m_jacCollision[2][5]=-self.m_puckRadius/3
                           self.m_jacCollision[3][3]=-self.m_e
                           self.m_jacCollision[4][5]=self.m_dt
                           self.m_jacCollision[5][2]=-2/(3*self.m_puckRadius)
                           self.m_jacCollision[5][5]=1/3
                           jacobian=self.m_rimGlobalTransformsInv[i]@self.m_jacCollision@self.m_rimGlobalTransforms[i]
                      else:
                           #velocity on next time step with sliding
                           slideDir=(vtScalar+dtheta*self.m_puckRadius)/abs(vtScalar+dtheta*self.m_puckRadius)
                           vtNextSCalar=vtScalar+self.m_rimFriction*slideDir*(1+self.m_e)*vnSCalar
                           vnNextScalar=-self.m_e*vnSCalar
                           state[5]=dtheta+2*self.m_rimFriction*slideDir*(1+self.m_e)*vnSCalar/self.m_puckRadius
                           self.m_jacCollision=np.eye(6)
                           self.m_jacCollision[0][2] = self.m_dt
                           self.m_jacCollision[1][3] = self.m_dt
                           self.m_jacCollision[2][3] = self.m_rimFriction*slideDir*(1+self.m_e)
                           self.m_jacCollision[3][3] = -self.m_e
                           self.m_jacCollision[4][5] = self.m_dt
                           self.m_jacCollision[5][3] = -2 / (3 * self.m_puckRadius)
                           jacobian = self.m_rimGlobalTransformsInv[i]@self.m_jacCollision@self.m_rimGlobalTransforms[i]
                      state[2:4]=vnNextScalar*vecN+vtNextSCalar*vecT
                      state[0:2]=p+s*u+(1-s)*state[0:2]*self.m_dt
                      state[4]=theta+s*dtheta*self.m_dt+(1-s)*state[5]*self.m_dt
                      return True
                 i+=1
            return False
















