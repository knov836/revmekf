import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numdifftools as nd 

from mpmath import mp
from mpmath import mpf
mp.dps = 40


from function_quat import *
from trajectory_gen import *




def test_wh(N,left,right,dw,radius,nticks):
    res_d = np.zeros(N)
    res_dthetas = np.zeros(N)
    for i in range(1,N):
        dl = left[i]*mpf(2*mp.pi*radius)/nticks
        dr = right[i]*mpf(2*mp.pi*radius)/nticks
        d = mpf(dl+dr)/mpf(2)
        dtheta = mpf(dr-dl)/mpf(2*dw)
        res_d[i] = d
        res_dthetas[i] = dtheta
    return res_d,res_dthetas

def test_orientation(N,left,right,dw,radius,nticks,quat,thetas,dthetas):
    
    orientation = np.zeros((N,4),dtype=mpf)
    theta = mpf(0)
    for i in range(1,N):
        dl = left[i]*mpf(2*mp.pi*radius)/nticks
        dr = right[i]*mpf(2*mp.pi*radius)/nticks
        d = mpf(dl+dr)/mpf(2)
        dtheta = mpf(dr-dl)/mpf(2*dw)
        orientation[i,0:3] =orientation[i-1,0:3] +np.array(quat_rot(np.array([mpf(0),d*mp.cos(theta+dtheta/mpf(2)),d*mp.sin(theta+dtheta/mpf(2)),mpf(0)],dtype=mpf), quat),dtype=mpf)[1:4]
        orientation[i,3] = orientation[i-1,3]+mpf(dtheta)
        theta = mpf(theta)+mpf(dtheta)
    return orientation


class KFilterDataSynth:
    def __init__(self, size,freq,g_bias=0,lw_noise = 0, rw_noise = 0,g_noise=0,a_noise = 0,m_noise=0,p_noise = 0,mode='GyroAccMag',traj='Rand',params_test=None,surf=np.array([-1,1,0]),dwheels=mpf(1.0),radius=mpf(1.0),nticks=mpf(10000.),delta=0.01):
        self.freq = freq
        self.size=size
        self.DT = mpf(1)/mpf(self.freq)
        normal = surf
        self.normal = normal
        self.surf = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
        self.gravity = np.array([0,0,1],dtype=mpf)
        
        self.mode = mode
        self.base_width=2*dwheels
        self.time = np.array([mpf(i)/mpf(freq) for i in range(size)])
        
        
        if self.mode =='GyroAccMag':
            if traj == 'Rand':
                self.pos_earth,self.orient,self.rotsurf = position_data_rand(size,self.surf,delta)
            elif traj == 'Rand3D':
                self.pos_earth,self.orient,self.rotsurf = position_data_rand_3D(size,delta)
            elif traj == 'Test0':
                
                self.pos_earth,self.orient,self.rotsurf = test_traj0(size,params_test['alpha'],self.surf)
        elif self.mode =='GyroAcc':
            if traj == 'Rand':
                self.pos_earth,self.orient,self.rotsurf = position_data_rand_smallyaw(size,self.surf)
            elif traj == 'Test0':
                
                self.pos_earth,self.orient,self.rotsurf = test_traj0(size,params_test['alpha'],self.surf)    
        elif self.mode == 'OdoAccPre':
            if traj == 'Rand':
                self.dw = dwheels
                self.radius = radius
                self.nticks = nticks
                self.pos_earth,self.orient,self.rotsurf,self.leftw,self.rightw,self.dthetas,self.thetas,self.dws = position_data_odo(size,self.surf,self.DT,dwheels,radius,nticks,delta)
                meter_per_ticks = mpf(2*mp.pi*radius)/mpf(nticks)
                #print(meter_per_ticks/self.DT)
                self.leftw = (self.leftw  )*meter_per_ticks/self.DT+2*(np.random.random_sample(size)-0.5)*lw_noise
                self.rightw = (self.rightw  )*meter_per_ticks/self.DT+2*(np.random.random_sample(size)-0.5)*rw_noise
                self.p_noise = p_noise
                self.pressure = self.pos_earth[:,2]+(np.random.random_sample(size)-0.5)*self.p_noise*2
        
        self.gyro = self.gyro_data(self.orient)
        self.speed_earth = self.calc_speed_earth(self.pos_earth)
        self.acc_earth = self.calc_acc_earth(self.speed_earth)
        self.mag_earth = self.calc_mag_earth()
        self.a_noise = a_noise
        self.m_noise = m_noise
        self.acc = self.mems_ref(self.acc_earth)+(np.random.random_sample((size,3))-0.5)*self.a_noise*2
        self.mag = self.mems_ref(self.mag_earth)+(np.random.random_sample((size,3))-0.5)*self.m_noise*2
        self.g_bias = 2*g_bias*(np.random.random_sample(3)-0.5)
        self.gyro +=2*(np.random.random_sample((size,3))-0.5)*g_noise+np.transpose(np.repeat(self.g_bias,size).reshape(3,size))
        self.neworient = self.new_orient()
        
        self.grav_earth = self.calc_grav_earth(self.speed_earth)
        self.gravs = self.mems_ref(self.grav_earth)
        self.mag0 = np.array([0,1,0],dtype=mpf)
        
        
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
            cc_t[i,0]=0
            cc_t[i,1:] = log_q(diffq)/dt    
            wx,wy,wz = cc_t[i,1:]
        return cc_t[:,1:]
    def calc_speed_earth(self,pos_earth):
        speed_ref = np.vstack([np.zeros(3,dtype=mpf),np.diff(pos_earth,axis=0)])/self.DT
        return speed_ref
    def calc_acc_earth(self,speed_earth):
        acc_earth = np.vstack([np.zeros(3,dtype=mpf),np.diff(speed_earth,axis=0)])/self.DT+np.array(self.gravity,dtype=mpf)

        acc= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            acc[i,:]=self.gravity

        acc[:acc_earth.shape[0]] = acc_earth
        return acc
    def calc_grav_earth(self,speed_earth):
        acc_earth= np.zeros((len(self.orient),3),dtype=mpf)+np.array(self.gravity,dtype=mpf)
        #acc_earth = np.vstack([np.zeros(3,dtype=mpf),np.diff(speed_earth,axis=0)])/self.DT+np.array(self.gravity,dtype=mpf)

        acc= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            acc[i,:]=self.gravity

        acc[:acc_earth.shape[0]] = acc_earth
        return acc
    
    def calc_mag_earth(self):
        mag= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            mag[i,1]=mpf(1)
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
            quat = quat_rot(dd_quat,conjbbi)
            res[i,:] = quat[1:4]
        return res
    
N=100
angle = int(N/2)
newset = KFilterDataSynth(N,mpf(0.1),mode='GyroAccMag') 
newset = KFilterDataSynth(N,mpf(0.1),mode='OdoAccPre') 

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.pos_earth)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient-newset.neworient)
ax.set_title("diff acc and computed orient")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc[:,0])


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient[:,0],newset.orient[:,3],"x")
ax.plot(newset.neworient[:,0],newset.neworient[:,3],"x")


tab = newset.mag[10,:]
R = QuatToRot(newset.neworient[10,:])

acc = newset.acc[1,:]
mag = newset.mag

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient)


if newset.mode == 'OdoAccPre':
    res_d,res_dthetas = test_wh(newset.size,newset.leftw,newset.rightw,newset.dw,newset.radius,newset.nticks)
    
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(res_d-newset.dws)
    ax.set_title("compare distance")
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(res_dthetas-newset.dthetas)
    ax.set_title("compare thetas")
    
    orientation = test_orientation(newset.size,newset.leftw,newset.rightw,newset.dw,newset.radius,newset.nticks,newset.rotsurf,newset.thetas,newset.dthetas)
    
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(orientation[:,:3]-newset.pos_earth[:,:3])
    ax.set_title("compare position")
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(newset.pos_earth[:,:3])
