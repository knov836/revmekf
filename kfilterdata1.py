
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numdifftools as nd 

from mpmath import mp
from mpmath import mpf
from function_quat import *

import sys,os





N = 100





def orient_data(N):
    orient = np.zeros((N,4),dtype=mpf)
    angle = 2*mp.pi
    alpha = int(50*N/100)
    for i in range(alpha):
        theta = mpf(i*angle)/mpf(N)
        #theta = 0
        orient[i,:] = [mp.cos(theta/2),0,0,mp.sin(theta/2)]
    for i in range(0,N-alpha):
        theta = mpf((alpha-i)*angle)/mpf(N)
        #theta = 0
        orient[alpha+i,:] = [mp.cos(theta/2),0,0,mp.sin(theta/2)]
    return orient



def position_data(N,alpha,Surface):
    pos_earth = np.array(np.zeros((N,3)),dtype=mpf)
    
    for i in range(5):
        pos_earth[i,0]=mpf(i)#/1000
        pos_earth[i,1]=mpf(i)#/1000
        pos_earth[i,2]=mpf(i)
    for i in range(5,10):
        pos_earth[i,0]=mpf(10-i)#/1000
        pos_earth[i,1]=mpf(10-i)#/1000
        pos_earth[i,2]=mpf(10-i)
    orient = np.zeros((N,4),dtype=mpf)
    angle = mpf(2)*mp.pi
    

    Surf = np.copy(Surface)
    Surface = Surface.astype(float)
    surf = lambda x : Surface[0] + Surface[1]*x[0]+ Surface[2]*x[1]+ Surface[3]*x[2]+ Surface[4]*x[0]*x[1]+ Surface[5]*x[1]*x[2]+ Surface[6]*x[2]*x[0] + Surface[7]*x[0]**2+ Surface[8]*x[1]**2+ Surface[9]*x[2]**2
    grad = nd.Gradient(surf) #
    
    normal1 = np.array([mpf(s) for s in Surf[1:4]],dtype=mpf)
    normal = np.array([0,0,1],dtype=mpf)
    n0,n1 =mp.norm(normal),mp.norm(normal1)

    newquat = skewSymmetric(normal)@normal1/(n1)


    #sign = np.sign(np.dot(newcross,))
    beta = mp.asin(mp.norm(newquat))
    if beta!=0:
        newquat = newquat/mp.sin(beta)*mp.sin(beta/2)
    else:
        newquat =newquat*mpf(0)
    Quaternion = np.zeros(4,dtype=mpf)
    Quaternion[1:4] = newquat
    Quaternion[0] = mp.cos(beta/2)
    Quat = np.copy(Quaternion)
    
    
    Quaternion = quat_ntom(normal,normal1)
    for i in range(N):
        if i<alpha:
            theta = mpf(i*angle)/mpf(N)
        else:
            theta = mpf((2*alpha-i)*angle)/mpf(N)
            #theta = 0
        #yaw_q = np.array([mp.cos(theta/2),mpf(0),mpf(0),mp.sin(theta/2)],dtype=mpf)
        
        dyaw_q = np.array(normal1*theta,dtype=mpf)
        yaw_q= ExpQua(dyaw_q)    
        
        position = project(pos_earth[i,:], surf,grad)
        pos_earth[i,:] = position
        
        
        qq = quat_inv(quat_mult((Quaternion),quat_inv(yaw_q)))

        orient[i,:] = normalize(qq)
    return pos_earth,orient,Quaternion

"""
POSITION
"""

"""
for i in range(5):
    pos_earth[i,0]=mpf(i)
    pos_earth[i,1]=mpf(i)
    pos_earth[i,2]=mpf(i)
"""    

"""
ORIENTATION
"""





grav=1




class KFilterDataSurf:
    def __init__(self, size,freq,alpha=0,g_bias=0,g_noise=0,a_noise = 0,m_noise=0,surf=np.array([-1,1,0])):
        self.freq = freq
        self.size=size
        self.DT = mpf(1)/mpf(self.freq)
        #self.grav = 1
        normal = surf
        self.normal = normal
        self.surf = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
        self.pos_earth,self.orient,self.rotsurf = position_data(size,alpha,self.surf)
        #self.orient = orient_data(size)
        self.grav = mpf(1)
        self.gravity = np.array([0,0,self.grav],dtype=mpf)
        
        
        self.gyro = self.gyro_data(self.orient)
        self.speed_earth = self.calc_speed_earth(self.pos_earth)
        self.acc_earth = self.calc_acc_earth(self.speed_earth)
        self.mag_earth = self.calc_mag_earth()
        self.a_noise = a_noise
        self.m_noise = m_noise
        self.acc = self.mems_ref(self.acc_earth)+(np.random.random_sample((size,3))-0.5)*self.a_noise*2
        self.omag = self.mems_ref(self.mag_earth)+(np.random.random_sample((size,3))-0.5)*self.m_noise*2
        self.mag = np.copy(self.omag)

        self.g_bias = 2*g_bias*(np.random.random_sample(3)-0.5)
        
        self.gyro +=2*(np.random.random_sample((size,3))-0.5)*g_noise+np.transpose(np.repeat(self.g_bias,size).reshape(3,size))
        
        
        self.neworient = self.new_orient()
        
        
    def gyro_data(self,orient):
        sample_period = mpf(1/self.freq)
        
        cc_t= np.zeros((len(orient),4),dtype=mpf)
        
        time = mpf(0)
        for i in range(1,len(orient)):
            dt = mpf(i)/self.freq - time
            time = mpf(i)/self.freq
            quat = (orient[i-1,:])
            conjbbi = np.array([quat[0],-quat[1],-quat[2],-quat[3]],dtype=mpf)
            quat = (orient[i,:])
            diffq = np.array(quat_mult(conjbbi,quat),dtype=mpf)
            #diffq = np.array(quat_mult(quat,conjbbi),dtype=mpf)
            cc_t[i,0]=0
            cc_t[i,1:] = log_q(diffq)/dt    
            #print(ExpQua((np.array(cc_t[i,1:]))*sample_period))
            wx,wy,wz = cc_t[i,1:]
        return cc_t[:,1:]
    def calc_speed_earth(self,pos_earth):
        speed_ref = np.vstack([np.zeros(3,dtype=mpf),np.diff(pos_earth,axis=0)])/self.DT
        return speed_ref
    def calc_acc_earth(self,speed_earth):
        acc_earth = np.vstack([np.zeros(3,dtype=mpf),np.diff(speed_earth,axis=0)])/self.DT+self.gravity

        acc= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            acc[i,:]=self.gravity

        acc[:acc_earth.shape[0]] = acc_earth
        return acc
    
    def calc_mag_earth(self):
        mag= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            mag[i,1]=1
        return mag
    def new_orient(self):
        new_orient = np.zeros((self.size,4),dtype=mpf)
        for i in range(self.size):
            a = self.acc[i,:]
            a = a/mp.norm(a)
            a0 = np.array([0,0,1],dtype=mpf)
            m = self.mag[i,:]
            m = m/mp.norm(m)
            m0 = np.array([0,1,0],dtype=mpf)
            M = np.array([-skewSymmetric(a)@m,m,a]).T
            new_orient[i,:] = normalize(quat_inv(RotToQuat(M)))
        return new_orient
    
    def mems_ref(self,tab):
        res = np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            
            dd_quat = np.array([0,tab[i,0],tab[i,1],tab[i,2]],dtype=mpf)
            conjbbi = np.array([-self.orient[i,0],self.orient[i,1],self.orient[i,2],self.orient[i,3]],dtype=mpf)
            #quat = quat_rot(dd_quat,conjbbi)
            quat = quat_rot(dd_quat,quat_inv(self.orient[i,:]))
            res[i,:] = quat[1:4]
        return res
    
N=100
angle = int(N/2)
newset = KFilterDataSurf(N,mpf(10.),alpha=angle,surf=np.array([0,0,1])) 

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.pos_earth)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient-newset.neworient)
ax.set_title("diff acc and computed orient")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)
ax.set_title("acc")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.mag)
ax.set_title("mag")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient[:,0],newset.orient[:,3],"x")
ax.plot(newset.neworient[:10,0],newset.neworient[:10,3],"x")


tab = newset.mag[10,:]
R = QuatToRot(newset.neworient[10,:])

acc = newset.acc[1,:]
mag = newset.mag

