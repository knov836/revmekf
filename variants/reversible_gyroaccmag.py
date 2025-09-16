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
    
    def linalg_correct(self,Gyroscope,Accelerometer,Magnetometer,Orient,normal=[0,0,1]):
        quat=self.rotsurf
        RR = QuatToRot((Orient))
        iRR = QuatToRot(quat_inv(Orient))
        
        logrot = log_q(self.Quaternion)
        
        nMagnetometer = np.array(quat_rot([0,*Magnetometer],self.Quaternion))[1:4]
        nAccelerometer = np.array(quat_rot([0,*Accelerometer],self.Quaternion))[1:4]
        
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
        acc1,rot1,irot1,acc2,rot2,irot2,acc3,rot3,irot3= acc_from_normal_imu(np.array(quat_rot([0,*np.array(nAxis,dtype=mpf)],quat))[1:4],np.array([0,1,0],dtype=mpf) , np.array(quat_rot([0,*(nAccelerometer*(self.dt**2))],quat))[1:4], normal, self.surf_center,start = logrot1,detection=self.detection)
        acc = np.array(quat_rot([0,*(acc1)],quat_inv(quat)),dtype=mpf)[1:4]
        rot = QuatToRot(quat_inv(quat))@rot1
        return acc
    
    def update(self,Gyroscope, Accelerometer,Magnetometer,Orient):
        self.Quaternion,self.Bias,self.Pk = update_tested(self.Quaternion, self.Bias, self.Pk, self.R, Gyroscope, Accelerometer, Magnetometer, Orient,mag0=np.array(self.mag0,dtype=mpf))
    
    def variant_update_f(self, Time, Surface, Accelerometer,Gyroscope, Magnetometer,Orient):
        self.predict(Gyroscope,Orient)
        mag = Magnetometer
        if self.detection:
            mag = np.array(quat_rot([0,0,1,0], quat_inv(self.Quaternion)))[1:4]
        
        grav_earth = self.linalg_correct(Gyroscope, Accelerometer, mag, Orient,normal=self.normal)
        self.gravity_r = grav_earth
        self.update(Gyroscope,grav_earth,Magnetometer,Orient)




