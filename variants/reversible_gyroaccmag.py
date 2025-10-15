import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
from mpmath import matrix

import numdifftools as nd 
from sympy.plotting import plot

import sympy as sym
import torch

from sympy import symbols, cos, sin
from sympy import Rational
from sympy import nsolve
from sympy import lambdify
from sympy import series
import scipy.optimize as opt
from .variant_gyroacc import Filter

from predict_test import predict as predict_tested
from update_test import update0 as update_tested
from function_quat import *
from proj_func import *
from linalg import *

mp.prec=40

class PredictFilter(Filter):
    def __init__(self, *args,**kwargs):#rate, Q, R, Pk, grav, quat, Bias,normal,base_width=mpf(1.0),proj_fun=None,rotsurf=None,time=mpf(0)):
        super().__init__(*args,**kwargs)#Q, R, Pk, grav, quat, Bias,normal,base_width=mpf(1.0),proj_fun=proj_fun,rotsurf=None,time=mpf(0))
        self.variant_update=self.variant_update_f
        self.corrected = False
        self.not_corrected = False
        self.label = ""
        self.angle = 0
    
    def linalg_correct(self,Gyroscope,Accelerometer,Magnetometer,Orient,normal=[0,0,1]):
        quat=self.rotsurf
        RR = QuatToRot((Orient))
        iRR = QuatToRot(quat_inv(Orient))
        
        logrot = log_q(self.Quaternion)
        
        nMagnetometer = np.array(quat_rot([0,*Magnetometer],self.Quaternion))[1:4]
        
        grav= np.copy(Accelerometer)
        #if np.abs(self.std_acc_z)>0.85:
        grav[2] = normal[2]*np.linalg.norm(self.gravity)
        #print(acc[2],Accelerometer[2],normal)
        #nAccelerometer = np.array(quat_rot([0,*acc],self.Quaternion))[1:4]
        
        nMagnetometer = Magnetometer
        nAccelerometer = Accelerometer
        
        
        mm0 = nMagnetometer
        
        nAxis = np.array([0,1,0],dtype=mpf)
        nAxis=nMagnetometer
        nAxis = nAxis/np.linalg.norm(nAxis)
        
        mm0 = nAxis
        
        mm1 = np.array([0,1,0],dtype=mpf)
        quat= quat_ntom(mm0,mm1)
        
        qq = quat_mult(self.Quaternion,quat_inv(quat))
        logrot1 = log_q(np.array(qq))
        #logrot1 = np.zeros(3)
        if self.neural:
            
            acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3,corrected,not_corrected,angle_acc,t0,t2,t3,t4,et0,et2,et3,et4= acc_from_normal_imu_grav_neural(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4],np.array(quat_rot([0,*(grav*(self.dt**2))],quat))[1:4], normal, self.surf_center,start = logrot1,heuristic=self.heuristic,correction = False)
            self.t0 = t0
            self.t2 = t2
            self.t3 = t3
            self.t4 = t4
            self.et0 = et0
            self.et2 = et2
            self.et3 = et3
            self.et4 = et4
            
            angles_last = np.array([t0,t2,t3,t4,et0,et2,et3,et4],dtype=np.float32)
            if len(self.xtensor)>0:
                if len(self.xtensor.shape)!=2:
                    angles_last_extra = angles_last[np.newaxis, :]
                    angles_last_extra_3D = np.repeat(angles_last_extra[:, np.newaxis, :], 20, axis=1)
                    X_input = np.concatenate([self.xtensor,angles_last_extra_3D], axis=2)
                    X_tensor_1 = torch.tensor(X_input, dtype=torch.float32)
                    outputs = self.model(X_tensor_1)
                    soft = torch.softmax(outputs, dim=1).numpy().squeeze()
                
                    p_class0 = soft[0]
                    p_class1 = soft[1]
                    #probs[i] = p_class1
                    correction = (p_class1 > 0.4) and (et3>=0) 
                    #correction = (p_class1 > 0.75) and (et3>=0) 
                    print("probas",p_class0,p_class1)
                    acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3,corrected,not_corrected,angle_acc,t0,t2,t3,t4,et0,et2,et3,et4= acc_from_normal_imu_grav_neural(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4],np.array(quat_rot([0,*(grav*(self.dt**2))],quat))[1:4], normal, self.surf_center,start = logrot1,heuristic=self.heuristic,correction = correction)
                    self.t0 = t0
                    self.t2 = t2
                    self.t3 = t3
                    self.t4 = t4
                    self.et0 = et0
                    self.et2 = et2
                    self.et3 = et3
                    self.et4 = et4
                else:
                    X_input = np.concatenate([self.xtensor.flatten(),angles_last]).flatten().reshape(1, -1)
                    probs_loaded  = self.model(X_input)
                    
                    print("probas",probs_loaded)
                    correction= (probs_loaded[0,1] > 0.5).astype(int)  and et3>=0
                    acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3,corrected,not_corrected,angle_acc,t0,t2,t3,t4,et0,et2,et3,et4= acc_from_normal_imu_grav_neural(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4],np.array(quat_rot([0,*(grav*(self.dt**2))],quat))[1:4], normal, self.surf_center,start = logrot1,heuristic=self.heuristic,correction = correction)
                    self.t0 = t0
                    self.t2 = t2
                    self.t3 = t3
                    self.t4 = t4
                    self.et0 = et0
                    self.et2 = et2
                    self.et3 = et3
                    self.et4 = et4
        else:
            if self.manual:
                acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3,corrected,not_corrected,label,angle_acc,t0,t2,t3,t4,et0,et2,et3,et4= acc_from_normal_imu_grav_manual(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4],np.array(quat_rot([0,*(grav*(self.dt**2))],quat))[1:4], normal, self.surf_center,start = logrot1,heuristic=self.heuristic,correction = self.correction)
                self.label=label
            else:
                
                acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3,corrected,not_corrected,angle_acc,t0,t2,t3,t4,et0,et2,et3,et4= acc_from_normal_imu_grav(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4],np.array(quat_rot([0,*(grav*(self.dt**2))],quat))[1:4], normal, self.surf_center,start = logrot1,heuristic=self.heuristic,correction = self.correction)
            self.t0 = t0
            self.t2 = t2
            self.t3 = t3
            self.t4 = t4
            self.et0 = et0
            self.et2 = et2
            self.et3 = et3
            self.et4 = et4
        if corrected:
            self.corrected=True
            #print(angle_acc)
            self.angle = angle_acc
        if not_corrected:
            self.not_corrected = True

        acc = np.array(quat_rot([0,*(acc1)],quat_inv(quat)),dtype=mpf)[1:4]
        rot = QuatToRot(quat_inv(quat))@rot1
        return acc
    
    def update(self,Gyroscope, Accelerometer,Magnetometer,Orient):
        self.Quaternion,self.Bias,self.Pk = update_tested(self.Quaternion, self.Bias, self.Pk, self.R, Gyroscope, Accelerometer, Magnetometer, Orient,mag0=np.array(self.mag0,dtype=mpf))
    
    def variant_update_f(self, Time, Surface, Accelerometer,Gyroscope, Magnetometer,Orient):
        self.predict(Gyroscope,Orient)
        self.corrected = False
        self.not_corrected = False
        self.angle = 0
        acc = np.copy(Accelerometer)
        mag = Magnetometer
        if self.heuristic:
            #print(Accelerometer,np.dot(self.gravity/np.linalg.norm(self.gravity),Surface[1:4]/np.linalg.norm(Surface[1:4])))
            #acc[2] = self.gravity[2]*np.dot(self.gravity/np.linalg.norm(self.gravity),Surface[1:4]/np.linalg.norm(Surface[1:4]))
            #print(np.dot(self.gravity/np.linalg.norm(self.gravity),Surface[1:4]/np.linalg.norm(Surface[1:4])))
            mag = np.array(quat_rot([0,1,0,0], quat_inv(self.Quaternion)))[1:4]
            mag0 =np.copy(self.mag0)
            mag0[2] = 0
            mag0 = mag0/np.linalg.norm(mag0)
            mag = np.array(quat_rot([0,*mag0], quat_inv(self.Quaternion)))[1:4]
            #mag=Magnetometer
        grav_earth = self.linalg_correct(Gyroscope, acc, mag, Orient,normal=self.normal)
        
        self.gravity_r = grav_earth
        self.update(Gyroscope,grav_earth,Magnetometer,Orient)




