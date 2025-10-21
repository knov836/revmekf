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

from scipy.spatial.transform import Rotation as R


mp.prec=40

class PredictFilter(Filter):
    def __init__(self, *args,**kwargs):#rate, Q, R, Pk, grav, quat, Bias,normal,base_width=mpf(1.0),proj_fun=None,rotsurf=None,time=mpf(0)):
        super().__init__(*args,**kwargs)#Q, R, Pk, grav, quat, Bias,normal,base_width=mpf(1.0),proj_fun=proj_fun,rotsurf=None,time=mpf(0))
        self.variant_update=self.variant_update_f
        self.corrected = False
        self.not_corrected = False
        self.label = ""
        self.angle = 0
    
    def roll_correct(self,Gyroscope,Accelerometer,Magnetometer,Orient,normal=[0,0,1]):
        quat=self.rotsurf
        RR = QuatToRot((Orient))
        iRR = QuatToRot(quat_inv(Orient))
        logrot = log_q(self.Quaternion)
        nMagnetometer = Magnetometer
        nAccelerometer = Accelerometer
        mm0 = nMagnetometer
        nAxis = np.array([0,1,0],dtype=mpf)
        nAxis=nMagnetometer
        nAxis = nAxis/np.linalg.norm(nAxis)
        
        mm0 = nAxis
        mm1 = np.array([0,1,0],dtype=mpf)
        quat= quat_ntom(mm0,mm1)
        
        mag0 = np.array(quat_rot([0,*Magnetometer],self.Quaternion))[1:4]
        mm2 = np.array(mag0,dtype=mpf)
        mag0 = np.copy(mag0)
        mag0[2] = 0
        mag0 = mag0/np.linalg.norm(mag0)
        quat2 = quat_ntom(mag0,mm1)
        qq = quat_mult(quat_mult(quat2,self.Quaternion),quat_inv(quat))
        logrot1 = log_q(np.array(qq))
        acc1= acc_from_normal_imu_roll(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4], np.array(quat_rot([0,*normal],(quat2)))[1:4], np.array(quat_rot([0,*self.surf_center],(quat2)))[1:4],start = logrot1,heuristic=self.heuristic,correction = self.correction)/(self.dt**2)
        acc = np.array(quat_rot([0,*(acc1)],quat_inv(quat)),dtype=mpf)[1:4]
        return acc
    
    def linalg_correct(self,Gyroscope,Accelerometer,Magnetometer,Orient,normal=[0,0,1]):
        quat=self.rotsurf
        RR = QuatToRot((Orient))
        iRR = QuatToRot(quat_inv(Orient))
        
        logrot = log_q(self.Quaternion)
        
        nMagnetometer = np.array(quat_rot([0,*Magnetometer],self.Quaternion))[1:4]
        
        grav= np.copy(Accelerometer)
        #if np.abs(self.std_acc_z)>0.85:
        #grav[2] = normal[2]*np.linalg.norm(self.gravity)
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
        
        mag0 = np.array(quat_rot([0,*Magnetometer],self.Quaternion))[1:4]
        print("mag0",mag0)
        mm2 = np.array(mag0,dtype=mpf)
        mag0 = np.copy(mag0)
        mag0[2] = 0
        mag0 = mag0/np.linalg.norm(mag0)
        quat2 = quat_ntom(mag0,mm1)
        """qq = quat_mult(self.Quaternion,quat_inv(quat))
        logq1 = log_q(np.array(self.Quaternion))
        qq0 = quat_ntom(logq1/np.linalg.norm(logq1),mm2)
        qq1 = quat_rot(self.Quaternion,qq0)
        qq2 = np.array(quat_mult(qq1,quat_ntom(self.mag0,mm1)))"""
        qq = quat_mult(quat_mult(quat2,self.Quaternion),quat_inv(quat))
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
                #pdb.set_trace()
                acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3,corrected,not_corrected,angle_acc,t0,t2,t3,t4,et0,et2,et3,et4= acc_from_normal_imu_grav(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4],np.array(quat_rot([0,*(grav*(self.dt**2))],quat))[1:4], np.array(quat_rot([0,*normal],(quat2)))[1:4], np.array(quat_rot([0,*self.surf_center],(quat2)))[1:4],start = logrot1,heuristic=self.heuristic,correction = self.correction)
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
        rot = QuatToRot(quat_inv(quat))@rot1@QuatToRot((quat2))
        #pdb.set_trace()
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
            """mag = np.array(quat_rot([0,0,1,0], quat_inv(self.Quaternion)))[1:4]
            acc = self.roll_correct(Gyroscope, acc, mag, Orient,normal=self.normal)"""
            mag0 =np.copy(self.mag0)
            mag0[2] = 0
            mag0 = mag0/np.linalg.norm(mag0)
            zaxis = np.array([0,0,1])
            #mag = np.array(quat_rot([0,*mag0],ExpQua(-zaxis*np.pi/2)))[1:4]
            mag = np.array(quat_rot([0,*mag], quat_inv(self.Quaternion)))[1:4]#.astype(float)
            acc = self.linalg_correct(Gyroscope, acc, mag, Orient,normal=self.normal).astype(float)
            
            
            #mag = np.array(quat_rot([0,*mag0], quat_inv(self.Quaternion)))[1:4]
            #mag=Magnetometer
        else:
            acc = self.linalg_correct(Gyroscope, acc, mag, Orient,normal=self.normal)
        mag = Magnetometer.astype(float)
        heading = np.arctan2(mag[1],mag[0])
        zaxis = np.array([0,0,1])
        
        normal = self.normal.astype(float)
        normal_h= np.array(quat_rot([0,*normal],ExpQua(-zaxis*heading)))[1:4]
        
        qq0 = quat_ntom(np.array([0,0,1]), normal_h)#-normal_h[0]*np.array([1,0,0]))#-normal[0]*np.array([1,0,0]))
        #pacc= acc/n.linalg.norm(acc)-np.dot(acc/n.linalg.norm(acc))
        """qq1 = quat_ntom(np.array(quat_rot([0,*acc/np.linalg.norm(acc)],qq0))[1:4],np.array([0,0,1]))
        logqq1 = np.dot(np.array(log_q(qq1)),normal)*normal*0
        revacc = np.array(quat_rot([0,0,0,1],quat_inv(quat_mult(ExpQua(logqq1),qq0))))[1:4].astype(float)"""
        revacc = np.array(quat_rot([0,0,0,1],quat_inv(qq0)))[1:4].astype(float)
        roll = np.arctan2(revacc[1],revacc[2])
        
        print("roll",revacc,Accelerometer)
        #pdb.set_trace()
        pitch = np.arctan2(-acc[0],np.sqrt(acc[2]**2+acc[1]**2))
        
        roth = R.from_euler('ZYX', [heading, pitch,roll], degrees=False)
        nacc = roth.inv().as_matrix()[:]@[0,0,1]
        nmag = roth.inv().as_matrix()[:]@self.mag0
        
        grav_earth=nacc
        self.gravity_r = grav_earth
        self.update(Gyroscope,grav_earth,nmag,Orient)




