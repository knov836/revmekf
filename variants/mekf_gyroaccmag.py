
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
        self.update(Gyroscope,Accelerometer,Magnetometer,Orient)
        
