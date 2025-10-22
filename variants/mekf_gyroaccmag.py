
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
import numdifftools as nd 

from .variant_gyroacc import Filter

from function_quat import *
from predict_test import predict as predict_tested
from update_test import update0 as update_tested
from proj_func import *
from scipy.spatial.transform import Rotation as R



class PredictFilter(Filter):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.variant_update=self.variant_update_f


    def update(self,Gyroscope, Accelerometer,Magnetometer,Orient):
        acc = Accelerometer
        mag=Magnetometer
        #print(self.gravity)
        #print(np.linalg.norm(self.gravity.astype(float)))
        #acc= np.array(quat_rot([0,0,0,1],quat_inv(self.Quaternion)))[1:4]
        #mag= np.array(quat_rot([0,*self.mag0],quat_inv(self.Quaternion)))[1:4]#*np.linalg.norm(self.gravity.astype(float))
        
        self.Quaternion,self.Bias,self.Pk = update_tested(self.Quaternion, self.Bias, self.Pk, self.R, Gyroscope, acc, mag, Orient,mag0=np.array(self.mag0,dtype=mpf))

    
    def variant_update_f(self, Time, Surface, Accelerometer,Gyroscope, Magnetometer,Orient):
        self.predict(Gyroscope,Orient)
        
        """mag = Magnetometer.astype(float)
        acc = Accelerometer
        heading = np.arctan2(mag[1],mag[0])
        zaxis = np.array([0,0,1])
        
        normal = self.normal.astype(float)
        normal_h= np.array(quat_rot([0,*normal],ExpQua(-zaxis*heading)))[1:4]
        
        qq0 = quat_ntom(np.array([0,0,1]), normal_h-normal_h[0]*np.array([1,0,0]))#-normal[0]*np.array([1,0,0]))
        revacc = np.array(quat_rot([0,0,0,1],quat_inv(qq0)))[1:4].astype(float)
        roll = np.arctan2(revacc[1],revacc[2])
        #pdb.set_trace()
        pitch = np.arctan2(-acc[0],np.sqrt(acc[2]**2+acc[1]**2))
        
        roth = R.from_euler('ZYX', [heading, pitch,roll], degrees=False)
        nacc = roth.inv().as_matrix()[:]@[0,0,1]
        nmag = roth.inv().as_matrix()[:]@self.mag0"""
        self.update(Gyroscope,Accelerometer,Magnetometer,Orient)
        
