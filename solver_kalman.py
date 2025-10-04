import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt

from mpmath import mp
from mpmath import mpf
from function_quat import *
from kalman_parameters import *
import sys,os
from scipy.spatial.transform import Rotation


mp.dps = 40


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/examples')[0])



class SolverFilterPlan:
    def __init__(self,kfilter,q0,q1,r0,r1,normal,newset,start=np.array([1,0,0,0],dtype=mpf),gravity=np.array([0,0,1],dtype=mpf),proj_fun=None,size=100,heuristic=False):
        """
        Parameters
        ----------
        kfilter : TYPE
            Kalman Filter.
        q0 : TYPE
            gyro noise.
        q1 : TYPE
            bias noise.
        r0 : TYPE
            acc noise.
        r1 : TYPE
            mag noise.
        normal : TYPE
            normal of the plan
        newset : TYPE
            synthetic data
        Returns
        -------
        None.

        """
        self.N = 2
        N = self.N
        
        self.newset = newset
        
        if not(self.newset == None):
            self.N = newset.size
            N = self.N
            self.mode = newset.mode
        else:
            self.mode = 'GyroAccMag'
        self.surface = np.concatenate(([0,],normal,np.zeros(6)))
        #surf = lambda x : Surface[0] + Surface[1]*x[0]+ Surface[2]*x[1]+ Surface[3]*x[2]+ Surface[4]*x[0]*x[1]+ Surface[5]*x[1]*x[2]+ Surface[6]*x[2]*x[0] + Surface[7]*x[0]**2+ Surface[8]*x[1]**2+ Surface[9]*x[2]**2
        if self.mode.startswith('GyroAccMag'):
            QQ,RR,PK = generate_kalman_parameters(q0,q1,r0,r1)
            self.QQ,self.RR,self.PK = QQ,RR,PK
        elif self.mode.startswith('GyroAcc'):
            QQ,RR,PK = generate_kalman_parameters_acconly(q0,q1,r0)
            self.QQ,self.RR,self.PK = QQ,RR,PK
            
        if self.mode.startswith('Odo'):
            QQ,RR,PK = generate_kalman_parameters_odo(q0,q1,r0,r1)
            self.QQ,self.RR,self.PK = QQ,RR,PK
        newset = self.newset
        bias = [0,0,0]
        RAD = (pi / 180)
        X = np.array(gravity,dtype=mpf)
        


        Quat = np.array(start,dtype=mpf)
        
        self.quaternion = np.zeros((N, 4),dtype=mpf)        
        self.gravity_r = np.zeros((N, 3),dtype=mpf)    
        self.orient = np.zeros((N, 4),dtype=mpf)
        self.orient[0,:] = Quat
        time0=0
        base_width=0
        self.rotsurf = RotSurf(normal)
        
        mag0 = np.array([0,1,0],dtype=mpf)
        
        if newset == None:
            Quat = np.array(start,dtype=mpf)
            
            dt= 0
        else:
            time0 = newset.time[0]
            Quat = newset.orient[0,:].flatten()
            Quat = start
            self.rotsurf = newset.rotsurf
            mag0 = newset.mag0
            #Quat = (np.array(quat_mult(newset.rotsurf,start)))
            #print(Quat)
            dt = mpf(1)/newset.freq
            X = np.array(newset.gravity,dtype=mpf)
            normal = newset.normal
            if self.mode.startswith('Odo'):
                base_width=newset.base_width
            self.orient = newset.orient
        self.gravity=X
        
        
        self.quaternion[0,:] = Quat
        


        self.biases = np.zeros((N,3),dtype=mpf)
        self.c_acc = np.zeros((N, 3),dtype=mpf)
        self.speed = np.zeros((N, 3),dtype=mpf)
        self.position = np.zeros((N, 3),dtype=mpf)

        
        self.KFilter = kfilter(dt, QQ,RR,PK, X,Quat, bias, normal,mag0=mag0,rotsurf=self.rotsurf,proj_fun=proj_fun,time=time0,base_width=base_width,heuristic=heuristic)
        self.gravity_r[0,:] = self.KFilter.gravity_r
        self.ind =0
    
    def update_noarg(self,time=-1):
        KFilter= self.KFilter
        N = self.N
        newset = self.newset
        Surface = self.surface
        i = self.ind
        if i>=N-1:
            return
        qq = (self.quaternion[i, :])
        if time==-1:
            time = mpf(i+1)/newset.freq
        
        if self.mode == 'GyroAccMag':
            KFilter.UpdateSensor(time,Surface, newset.acc[i+1, :],newset.gyro[i+1, :],newset.mag[i+1,:],newset.orient[i+1,:])
        elif self.mode == 'GyroAcc':
            KFilter.UpdateSensor(time, Surface,newset.acc[i+1, :],newset.gyro[i+1, :],newset.orient[i+1,:])
        elif self.mode == 'OdoAccPre':
            KFilter.UpdateSensor(time,Surface,newset.acc[i+1, :], newset.pressure[i+1],newset.leftw[i+1],newset.rightw[i+1],self.orient[i+1,:])
        self.quaternion[i+1, :] = KFilter.Quaternion[0:4]
        self.gravity_r[i+1, :] = KFilter.gravity_r[0:4]
        
        self.biases[i+1, :] = KFilter.Bias[0:3]
        self.c_acc[i+1, :] = KFilter.acc
        self.speed[i+1, :] = KFilter.speed
        
        self.position[i+1, :] = KFilter.position
        self.ind = i+1
    def inc(self):
        self.ind = self.ind+1
        if self.ind>=self.N-1:
            self.pad()
    def update(self,time,*args,**kargs):#gyro,acc,mag,normal):
        KFilter= self.KFilter
        N = self.N
        newset = self.newset#self.newset
        
        i = self.ind
        
            
        
        qq = (self.quaternion[i, :])
        if newset==None:
            KFilter.UpdateSensor(mpf(time),Surface,acc,gyro, mag,self.orient[i,:])
        else:
            """if self.mode.startswith('GyroAccMag'):
                KFilter.UpdateSensor(mpf(time),Surface, acc,gyro,mag,self.orient[i,:])
            elif self.mode.startswith('GyroAcc'):
                KFilter.UpdateSensor(mpf(time),Surface, acc,gyro,self.orient[i,:])
            elif self.mode.startswith('Odo'):
                KFilter.UpdateSensor(mpf(time),Surface,newset.acc[i+1, :], newset.pressure[i+1],newset.leftw[i+1],newset.rightw[i+1], self.orient[i+1,:])"""

            if time==-1:
                time = mpf(i+1)/newset.freq
            
            if self.mode == 'GyroAccMag':
                gyro, acc, mag,normal= args[:4]
                std_acc_z = 0
                if "std_acc_z" in kargs.keys():
                    std_acc_z = kargs["std_acc_z"]
                Surface = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
                KFilter.UpdateSensor(time,Surface, acc,gyro,mag,newset.orient[i,:],std_acc_z = std_acc_z)
            elif self.mode == 'GyroAcc':
                gyro, acc, normal= args[:3]
                Surface = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
                KFilter.UpdateSensor(time, Surface,newset.acc[i+1, :],newset.gyro[i+1, :],newset.orient[i+1,:])
            elif self.mode == 'OdoAccPre':
                acc,pressure,leftw,rightw= args[:4]
                Surface = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
                KFilter.UpdateSensor(time,Surface,acc, pressure,leftw,rightw,self.orient[i,:])
        self.gravity_r[i+1, :] = KFilter.gravity_r[0:4]
        
        #KFilter.UpdateSensor(mpf(time),gyro, acc,mag,None,Surface)
        
        
        self.quaternion[i+1, :] = KFilter.Quaternion[0:4]
        self.biases[i+1, :] = KFilter.Bias[0:3]
        self.c_acc[i+1, :] = KFilter.acc
        self.speed[i+1, :] = KFilter.speed
        
        self.position[i+1, :] = KFilter.position

        self.inc()
            
    def pad(self):
        self.N=self.N+1
        self.quaternion = np.pad(self.quaternion,[(0,1),(0,0)])
        self.orient = np.pad(self.orient,[(0,1),(0,0)])
        self.c_acc= np.pad(self.c_acc,[(0,1),(0,0)])
        self.biases= np.pad(self.biases,[(0,1),(0,0)])
        self.position= np.pad(self.position,[(0,1),(0,0)])
        self.speed= np.pad(self.speed,[(0,1),(0,0)])
    def get_attitude(self):
        return self.quaternion[self.ind,:]
    def get_pos(self):
        return self.position[self.ind,:]
    def solve(self):
        KFilter= self.KFilter
        N = self.N
        newset = self.newset
        Surface = self.surface
        for i in range(self.ind,N-1):
        #for i in range(2):
            qq = (self.quaternion[i, :])
            KFilter.UpdateSensor(mpf(i+1)/newset.freq,Surface, newset.acc[i+1, :],newset.gyro[i+1, :],newset.mag[i+1,:],newset.orient[i+1,:])
            #print(qq)
            #print("solving")
            
            self.quaternion[i+1, :] = KFilter.Quaternion[0:4]
            #print(self.quaternion[i+1, :])
            self.biases[i+1, :] = KFilter.Bias[0:3]
            self.c_acc[i+1, :] = KFilter.acc
            self.speed[i+1, :] = KFilter.speed
            
            self.position[i+1, :] = KFilter.position
            
