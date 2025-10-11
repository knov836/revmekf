
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
import numdifftools as nd 


from function_quat import *
from predict_test import predict as predict_tested
from update_test import update0 as update_tested
from proj_func import *


class Filter:
    def __init__(self, rate, Q, R, Pk, grav, quat,Bias,normal,mag0=np.array([0,1,0],dtype=mpf),rotsurf=np.zeros(4,dtype=mpf),proj_fun=None,time=mpf(0),base_width=mpf(1.0),heuristic=False,neural=False,manual=False):
        self.proj_fun = proj_fun
        self.rate = rate
        self.Bias = np.array(Bias,dtype=mpf)
        self.Quaternion = np.array(quat,dtype=mpf)
        
        self.Quaternion[0:4] = self.Quaternion[0:4]/mp.norm(self.Quaternion)
        self.acc = np.zeros(3,dtype=mpf)
        self.speed = np.zeros(3,dtype=mpf)
        self.position = np.zeros(3,dtype=mpf)
        self.Q = np.array(Q).reshape(6,6)
        self.R = np.array(R)
        self.Pk = np.array(Pk,dtype=mpf).reshape(6,6)
        self.time = time
        self.amp = 0
        self.angle=0
        self.state = np.zeros(6,dtype=mpf)
        self.rotsurf = np.array(rotsurf,dtype=mpf)
        self.quat = np.array(quat_mult(quat_inv(self.rotsurf),quat_inv(self.Quaternion)),dtype=mpf)[[0,3]]
        self.normal = np.array(normal,dtype=mpf)
        self.center= np.zeros(3,dtype=mpf)
        self.surf_center= np.zeros(3,dtype=mpf)
        self.gravity = np.array(grav,dtype=mpf)
        self.gravity_r = np.array(quat_rot([0,*(grav)],quat_inv(self.Quaternion)))[1:4]
        self.grav_earth = np.copy(self.gravity)
        self.mag0 = mag0
        self.heuristic=heuristic
        self.manual=manual
        self.neural=neural
        self.correction = False
        self.std_acc_z = 0
        self.variant_update = None
        self.t0 = 0
        self.t4 = 0
        self.t2 = 0
        self.t3 = 0
        self.et0 = 0
        self.et4 = 0
        self.et2 = 0
        self.et3 = 0
        
        
        
    def predict_speed(self,acc):
        self.speed = self.speed +acc*self.dt
    def predict_sspeed(self):
        return self.speed
    def predict_position(self):
        self.position = self.position +self.speed*self.dt
    def predict_position_speed(self,speed):
        self.position = self.position +speed*self.dt
    def predict_position_acc(self,acc):
        self.position = self.position +acc*(self.dt)**2
    #def test_predict_position_acc(self,acc):
    #    print("test predict pos acc",self.position +acc*(self.dt)**2)
    def predict_sposition(self):
        self.s_position = self.s_position +self.s_speed*self.dt
        
    def predict(self,Gyroscope,Orient):
        self.Quaternion,self.Bias,self.Pk,Phi = predict_tested(self.Quaternion, self.Bias, self.Pk, self.Q, Gyroscope, Orient,dt=self.dt)
    def compute_center(self,Surface):
        """
        Center : affine point, given the position, projection of the position on the surface

        Parameters
        ----------
        Surface : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        cst = Surface[0]
        normal = Surface[1:4]
        normal = normal/mp.norm(normal)
        self.center= self.position-np.dot(normal,self.position)*normal+cst*normal #+/- cst ???
        self.surf_center= -np.dot(normal,self.position)*normal+cst*normal

    def UpdateSensor(self, *args,**kargs):
        Time, Surface, Accelerometer = args[:3]
        self.dt = mpf(Time) - self.time
        self.time = mpf(Time)
        
        normal = Surface[1:4]
        normal = normal/mp.norm(normal)
        self.normal = normal
        
        normal0 = np.array([0,0,1],dtype=mpf)
        self.rotsurf = quat_ntom(normal0,normal)
        speed = self.predict_sspeed()
        self.predict_position_speed(speed)
        Surface[0] =  np.dot(self.position,normal)
        self.predict_speed(-self.gravity)
        self.predict_position_acc(-self.gravity)
        self.compute_center(Surface)
        self.acc = Accelerometer
        if kargs["std_acc_z"]!= None:
            self.std_acc_z = kargs["std_acc_z"]
        if kargs["correction"]!= None:
            self.correction= kargs["correction"]
        
        if self.variant_update !=None:
            self.variant_update(*args)
        
        
        
        ax,ay,az = Accelerometer
        if self.proj_fun == None:
            acc_earth = np.array(quat_rot([0,ax,ay,az],self.Quaternion)[1:4],dtype=mpf)
        else:
            Surface[0] = 0
            self.compute_center(Surface)
            self.Quaternion,self.Pk,acc_earth = self.proj_fun(self.dt, self.Quaternion, self.Pk, self.position, self.center, Accelerometer, Surface)
        
        self.predict_position_acc(acc_earth)
        old_speed = self.speed
        self.predict_speed(acc_earth)
        #print(self.position.dot(normal))
