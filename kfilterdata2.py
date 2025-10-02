
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numdifftools as nd 

from mpmath import mp
from mpmath import mpf
from function_quat import *
import pandas as pd
import sys,os



grav=1



class KFilterDataFile:
    def __init__(self, data,mode='GyroAccMag',g_bias = np.array([0,0,0],dtype=mpf),base_width=mpf(1.0),surf=np.array([-1,1,0],dtype=mpf),normal=np.array([]),gravity=np.array([])):
        gyro = np.array(data[:,4:7],dtype=mpf)#/180*np.pi
        acc = np.array(data[:,1:4],dtype=mpf)
        mag = np.array(data[:,7:10],dtype=mpf)
        time = np.array(data[:,0],dtype=mpf)/10**9
        self.base_width=base_width
        self.mode = mode
        self.normal = normal
        dtime = np.diff(time)
        #print(dtime)
        
        ind = np.where(1/dtime<1000)[0]
        #print(data[])
        
        self.freq = np.mean(1/dtime[ind])
        self.size=len(time)
        self.c_size = 150
        self.DT = mpf(1)/mpf(self.freq)
        self.grav = np.mean([mp.norm(a) for a in acc])
        
        
        
        self.time = time
        #self.gyro = gyro
        self.pos_earth = np.zeros((self.size,3))
        self.speed_earth = np.zeros((self.size,3))
        self.acc_earth = np.zeros((self.size,3))
        self.mag_earth = np.zeros((self.size,3))
        self.a_noise = 0
        self.m_noise = 0
        self.acc = acc
        self.mag = mag
        self.g_bias = np.mean(gyro[:self.c_size,:],axis=0)
        self.gyro = np.copy(gyro-self.g_bias)
        self.c_mag = np.copy(self.cmag(normal=self.normal))
        calib = self.new_orient_calib()
        quat_calib = np.mean(calib[:self.c_size,:],axis=0)
        
        self.quat_calib = quat_calib/np.linalg.norm(quat_calib)
        
        if len(self.normal)==0:
            self.normal = np.array(quat_rot([0,0,0,1],(self.quat_calib)))[1:4]
            normal = self.normal
            
        m_mag = np.mean(mag[:self.c_size,:],axis=0)
        
        self.d_mag = np.array(quat_rot([0,*m_mag],self.quat_calib))[1:4]
        self.d_mag = self.d_mag/np.linalg.norm(self.d_mag)
        oo = skewSymmetric(self.normal)@self.d_mag
        plane_mag = oo/np.linalg.norm(oo)
        self.plane_mag = plane_mag
        
        new_normal = skewSymmetric(self.d_mag)@plane_mag
        self.new_normal = new_normal
        east = np.array([0,1,0],dtype=mpf)
        orth = skewSymmetric(self.normal)@east
        orth=orth/np.linalg.norm(orth)
        
        target_vector = -skewSymmetric(new_normal)@orth
        target_vector = target_vector/np.linalg.norm(target_vector)
        
        qq0 = quat_ntom(self.d_mag, target_vector)
        beta_rot_mag = log_q(qq0)
        self.beta_mag = np.linalg.norm(beta_rot_mag)*np.sign(np.dot(new_normal,beta_rot_mag))
        
        assert(np.linalg.norm(beta_rot_mag- self.beta_mag*new_normal)<10**(-10))
        
        r_plane_mag = east
        self.orth = orth
        self.target_vector = target_vector
        self.r_plane_mag = r_plane_mag
        l_rot_mag = log_q(quat_ntom(target_vector, east))
        self.alpha_mag = np.linalg.norm(l_rot_mag)*np.sign(np.dot(orth,l_rot_mag))
        assert(np.linalg.norm(l_rot_mag - self.alpha_mag*orth)<10**(-10))
        
        self.mag = np.copy(self.omag(self.normal))
        self.neworient = self.new_orient()
            
        self.rotsurf = quat_ntom(np.array([0,0,1]),normal)
        self.surf = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
        
        
        
        self.orient = np.copy(self.neworient)
        self.orient[0,:] = self.quat_calib
        
        perceived_gravity = -np.mean(self.acc[:self.c_size,:],axis=0)
        self.pg_std = np.std(self.acc[:100,:].astype(float),axis=0)
        n_perceived_gravity = perceived_gravity/np.linalg.norm(perceived_gravity)
        qq = quat_ntom(n_perceived_gravity, np.array([0,0,1],dtype=mpf))
        if len(gravity) == 0:
            self.gravity = np.array(quat_rot([0,*perceived_gravity],qq))[1:4]#+0.01#-np.array([0,0,np.linalg.norm(self.pg_std)],dtype=mpf)/10
        else:
            self.gravity=gravity
        self.mag0 = np.array(quat_rot([0,*np.mean(self.mag[:300,:].astype(float),axis=0)], self.quat_calib))[1:4]
        self.mag0 = np.array([0,1,0],dtype=mpf)
        
    def cmag(self,normal=None):
        cmag= np.zeros((self.c_size,3),dtype=mpf)
        for i in range(self.c_size):
            a=self.acc[i,:]
            a=a/np.linalg.norm(a)
            m=self.mag[i,:]
            if mp.norm(m)!=0:
                m=m/mp.norm(m)
            cmag[i,:] = m
            #continue
            if mp.norm(m)!=0:
                m=m/mp.norm(m)
                adm = skewSymmetric(a)@m
                adm = adm/mp.norm(adm)
                new_m = skewSymmetric(adm)@a
                cmag[i,:] = new_m
            #print(m,new_m)
        
        return cmag
    def omag(self,normal):
        omag= np.zeros((self.size,3),dtype=mpf)
        for i in range(self.size):
            
            m=self.mag[i,:]
            if mp.norm(m)!=0:
                m=m/mp.norm(m)
            omag[i,:] = m
            #continue
            if mp.norm(m)!=0:
                a=self.acc[i,:]
                a = a/np.linalg.norm(a)            
                m=m/mp.norm(m)
                adm = skewSymmetric(a)@m
                adm = adm/mp.norm(adm)
                new_new_m= skewSymmetric(adm)@a
                
                omag[i,:] = new_new_m
        
        return omag

    
    def calc_speed_earth(self,pos_earth):
        speed_ref = np.vstack([np.zeros(3,dtype=mpf),np.diff(pos_earth,axis=0)])/self.DT
        return speed_ref
    def calc_acc_earth(self,speed_earth):
        acc_earth = np.vstack([np.zeros(3,dtype=mpf),np.diff(speed_earth,axis=0)])/self.DT+np.array(self.gravity)

        acc= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            acc[i,2]=1

        acc[:acc_earth.shape[0]] = acc_earth
        return acc
    
    def calc_mag_earth(self):
        mag= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            mag[i,1]=1
        return mag
    
    def new_orient_calib(self):
        new_orient = np.zeros((self.c_size,4),dtype=mpf)
        for i in range(self.c_size):
            a = self.acc[i,:]
            a = a/mp.norm(a)
            a0 = np.array([0,0,1],dtype=mpf)
            #a = -a0
            m = self.c_mag[i,:]
            m = m/mp.norm(m)
            m0 = np.array([0,1,0],dtype=mpf)
            
            adm = skewSymmetric(a)@m
            adm = adm/mp.norm(adm)
            new_m = skewSymmetric(adm)@a
            #new_m = m
            M = np.array([-adm,new_m,a]).T
            #print("new_m",new_m,m)
            #M = np.array([new_m,adm,a]).T
            new_orient[i,:] = normalize(quat_inv(RotToQuat(M)))
        return new_orient
    def new_orient(self):
        new_orient = np.zeros((self.size,4),dtype=mpf)
        for i in range(self.size):
            a = self.acc[i,:]
            a = a/mp.norm(a)
            a0 = np.array([0,0,1],dtype=mpf)
            #a = -a0
            m = self.mag[i,:]
            m = m/mp.norm(m)
            m0 = np.array([0,1,0],dtype=mpf)
            adm = skewSymmetric(a)@m
            adm = adm/mp.norm(adm)
            new_m = skewSymmetric(adm)@a
            
            M = np.array([-adm,new_m,a]).T
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



data_file = 'test1.csv'
data_file = 'calibration_data_250523.csv'
data_file = 'imu_data_sbg.csv'
mag_file = 'mag_data_sbg.csv'
#N = 10000
data=pd.read_csv(data_file)
mag=pd.read_csv(mag_file)
n_start = 0
n_end=1200
df = data.values[n_start:n_end,:]
c_mag = mag.values[n_start:n_end,:]

newset = KFilterDataFile(df[1:,:],surf=np.array([0,0,1])) 

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.pos_earth)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient-newset.neworient)
ax.set_title("diff acc and computed orient")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc[:,:])
ax.set_title("acc")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.mag[:,:])
ax.set_title("mag")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(c_mag[:,:])
ax.set_title("c_mag")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient[:,0],newset.orient[:,3],"x")
ax.plot(newset.neworient[:,0],newset.neworient[:,3],"x")
ax.set_title("compare orient")

R = QuatToRot(newset.neworient[10,:])
