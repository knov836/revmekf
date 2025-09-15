
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
    def __init__(self, data,mode='OdoAccPre',g_bias = np.array([0,0,0],dtype=mpf),base_width=mpf(1.0),surf=np.array([-1,1,0],dtype=mpf)):
        leftw = np.array(data[:,1],dtype=mpf)#/180*np.pi
        rightw = np.array(data[:,2],dtype=mpf)
        acc = np.array(data[:,3:6],dtype=mpf)
        pressure = np.array(data[:,6],dtype=mpf)
        time = np.array(data[:,0],dtype=mpf)/10**9
        self.base_width=base_width
        self.mode = mode
        dtime = np.diff(time)
        
        ind = np.where(1/dtime<1000)[0]
        self.freq = np.mean(1/dtime[ind])
        self.size=len(time)
        self.DT = mpf(1)/mpf(self.freq)
        self.grav = np.mean([mp.norm(a) for a in acc])
        normal = surf
        self.normal = surf
        
        self.time = time
        self.pos_earth = np.zeros((self.size,3))
        
        
        self.leftw = leftw#np.diff(leftw)
        self.rightw = rightw#np.diff(rightw)
        self.pressure = pressure-pressure[0] #Start at zero!
        self.acc = acc
        print(data.shape)
        calib = self.new_orient_calib()
        
        
        
        quat_calib = np.mean(calib[:50,:],axis=0)
        
        self.quat_calib = quat_calib/np.linalg.norm(quat_calib)
        self.rotsurf = quat_ntom(np.array([0,0,1]),normal)
        self.surf = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
        
        self.orient = np.zeros((self.size,4))
        self.orient[0,:] = self.quat_calib
        
        self.mag0 = np.array([0,1,0],dtype=mpf)
        perceived_gravity = -np.mean(self.acc[:50,:],axis=0)
        n_perceived_gravity = perceived_gravity/np.linalg.norm(perceived_gravity)
        qq = quat_ntom(n_perceived_gravity, np.array([0,0,1],dtype=mpf))
        self.gravity = np.array(quat_rot([0,*perceived_gravity],qq))[1:4]
        
    def omag(self):
        omag= np.zeros((self.size,3),dtype=mpf)
        for i in range(self.size):
            a=self.acc[i,:]
            a = a/mp.norm(a)
            m=self.mag[i,:]
            m=m/mp.norm(m)
            adm = skewSymmetric(a)@m
            adm = adm/mp.norm(adm)
            new_m = skewSymmetric(adm)@a
            omag[i,:] = new_m
        return omag

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
        new_orient = np.zeros((self.size,4),dtype=mpf)
        for i in range(self.size):
            a = self.acc[i,:]
            a = a/mp.norm(a)
            a0 = np.array([0,0,1],dtype=mpf)
            m = np.array([0,0,1],dtype=mpf)
            m = np.array([-1,0,0],dtype=mpf)
            
            m0 = np.array(self.normal,dtype=mpf)
            adm = skewSymmetric(a)@m
            adm = adm/mp.norm(adm)
            new_m = skewSymmetric(adm)@a
            M = np.array([new_m,adm,a]).T
            new_orient[i,:] = (quat_inv(RotToQuat(M)))
        return new_orient
    def new_orient(self):
        new_orient = np.zeros((self.size,4),dtype=mpf)
        for i in range(self.size):
            a = self.acc[i,:]
            a = a/mp.norm(a)
            a0 = np.array([0,0,1],dtype=mpf)
            m = self.mag[i,:]
            m = m/mp.norm(m)
            m0 = np.array([0,1,0],dtype=mpf)
            adm = skewSymmetric(a)@m
            adm = adm/mp.norm(adm)
            new_m = skewSymmetric(adm)@a
            self.mag[i,:] = new_m
            
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
