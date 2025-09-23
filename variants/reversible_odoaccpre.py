
#import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
import numpy

import numdifftools as nd 

from .variant_odoacc import Filter
from function_quat import *
from predict_odo_test import predict as predict_tested
from update_odo_test import update0 as update_tested
from proj_func import *
from linalg import *
import numpy as np


class PredictFilter(Filter):
    def __init__(self, *args,**kwargs):#rate, Q, R, Pk, grav, quat, Bias,normal,base_width=mpf(1.0),proj_fun=None,rotsurf=None,time=mpf(0)):
        super().__init__(*args,**kwargs)#Q, R, Pk, grav, quat, Bias,normal,base_width=mpf(1.0),proj_fun=proj_fun,rotsurf=None,time=mpf(0))
        self.variant_update=self.variant_update_f
    
    def linalg_correct(self,Accelerometer,Pressure,Surface,Orient):
        #print("update speed",self.Quaternion, self.Bias, self.Pk, self.R, Gyroscope, Accelerometer, Magnetometer)
        quat=self.rotsurf
        #acc = np.array(quat_rot(quat_rot([0,*Accelerometer],quat_inv(quat)),quat_inv(quat)))[1:4]
        
        logrot = log_q(np.array(quat_mult(quat,ExpQua(self.state[3:6]))))
        qq = quat_inv(quat)
        
        
        mm0 = np.array(Surface[1:4],dtype=mpf)
        mm0 = mm0/mp.norm(mm0)
        
        mm1 = np.array([0,1,0],dtype=mpf)
        quat0= quat_ntom(mm0,mm1)
        #quat0 = quat_inv(quat)
        Quaternion = quat_mult(quat,ExpQua(self.state[3:6]))
        #logrot1 = log_q(np.array(quat_rot(quat_mult(quat_mult(quat,ExpQua(self.state[3:6])),(qq)),(quat0))))
        #logrot1 = np.zeros(3)
        
        qq1 = quat_mult(Quaternion,quat_mult(qq,quat_inv(quat0)))
        #logrot1 = log_q(np.array(qq1))
        qq2 = quat_mult(quat,ExpQua(self.state[3:6]))
        #print(quat_rot([0,0,0,1], quat_inv(qq2)),Accelerometer,mm0)
        acc = Accelerometer.astype(float)
        acc_angle = np.arctan2(acc[0],acc[1])
        nstate = np.array([0,0,acc_angle])
        logrot1 = log_q(np.array(quat_rot(quat_mult(quat_mult(quat,ExpQua(self.state[3:6])),(qq)),(quat0))))
        #acc1,rot1,irot1= acc_from_normal1(mm1,mm1 , np.array(quat_rot(quat_rot([0,*(Accelerometer*(self.dt**2))],quat_inv(qq)),quat0))[1:4], np.array(quat_rot([0,*np.array([0,0,1])],quat0))[1:4], np.array(quat_rot([0,*self.surf_center],quat0))[1:4],start = logrot1,s_rot=self.state[3:6])
        acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3= acc_from_normal1(mm1,mm1 , np.array(quat_rot(quat_rot([0,*(Accelerometer*(self.dt**2))],quat_inv(qq)),quat0))[1:4], np.array(quat_rot([0,*np.array([0,0,1])],quat0))[1:4], np.array(quat_rot([0,*self.surf_center],quat0))[1:4],start = logrot1,s_rot=self.state[3:6],heuristic=self.heuristic)
        #acc1,rot1,irot1= acc_from_mag(mm1,mm1 , np.array(quat_rot(quat_rot([0,*(Accelerometer*(self.dt**2))],quat_inv(qq)),quat0))[1:4], np.array(quat_rot([0,*np.array([0,0,1])],quat0))[1:4], np.array(quat_rot([0,*self.surf_center],quat0))[1:4],start = logrot1)
        #print(acc1,rot1@np.array([0,0,1],dtype=mpf))
        #q1 = RotToQuat(rot1)
        #q2 = quat_rot(q1, quat_inv(quat0))
        iquat0 = quat_inv(quat0)
        R1 = QuatToRot(iquat0)@rot1@QuatToRot(quat0)
        #acc1 = np.array(quat_rot(np.array([0,0,0,1],dtype=mpf),q2))[1:4]
        acc1 = R1 @ np.array([0,0,1],dtype=mpf)
        
        R2 = QuatToRot(iquat0)@rot2@QuatToRot(quat0)
        acc2 = R2 @ np.array([0,0,1],dtype=mpf)
        
        R3 = QuatToRot(iquat0)@rot3@QuatToRot(quat0)
        acc3 = R3 @ np.array([0,0,1],dtype=mpf)
        #irot1 = rot1.inv()
        #rot3 = (QuatToRot(qq)@rot1)**(-1)
        #irot3 = rot3**(-1)
        acc1_f = np.array(quat_rot([0,*(acc1)],qq),dtype=mpf)[1:4]
        acc2_f = np.array(quat_rot([0,*(acc2)],qq),dtype=mpf)[1:4]
        acc3_f = np.array(quat_rot([0,*(acc3)],qq),dtype=mpf)[1:4]
        
        acc = acc1_f
        """if np.linalg.norm(acc2_f-self.gravity_r)<np.linalg.norm(acc-self.gravity_r):
            acc = acc2_f
        if np.linalg.norm(acc3_f-self.gravity_r)<np.linalg.norm(acc-self.gravity_r):
            acc = acc3_f"""
        #acc = np.array(quat_rot([0,*acc],quat_inv(quat0)))[1:4]
        #acc =acc2
        #rot = 
        
        #acc0 = np.array(quat_rot([0,0,0,1],quat_inv(quat)))[1:4]
        #old_state = np.copy(self.state)
        #iRR = QuatToRot(quat_inv(Orient))
        return acc
    
    def update(self,Accelerometer,Surface):
        
        quat=self.rotsurf
        acc0 = np.array(quat_rot([0,0,0,1],quat_inv(quat)))[1:4]
        acc = Accelerometer
        self.state,self.Pk = update_tested(self.state, self.Pk, self.R, acc,acc0, Surface)
        
    def variant_update_f(self,Time, Surface,Accelerometer,Pressure, leftw, rightw,Orient):#(self, Time, Surface, Accelerometer,Gyroscope, Magnetometer,Orient):
        leftw=leftw*self.dt
        rightw=rightw*self.dt
        self.predict(leftw,rightw,Surface)
        #coef = 1#self.gravity_r[2]/Accelerometer[2]
        Accelerometer = self.relative_step#*coef
        grav_earth = self.linalg_correct(Accelerometer,Pressure,Surface,Orient)
        self.gravity_r = grav_earth
        self.update(grav_earth,Surface)
        #self.update(Accelerometer,Surface)