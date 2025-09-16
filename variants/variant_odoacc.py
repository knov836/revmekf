
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
from predict_odo_test import predict as predict_tested
from update_test import update0 as update_tested
from proj_func import *


class Filter:
    def __init__(self, rate, Q, R, Pk, grav, quat,Bias,normal,mag0=np.array([0,1,0],dtype=mpf),rotsurf=np.zeros(4,dtype=mpf),proj_fun=None,time=mpf(0),base_width=mpf(1.0),detection=False):
        self.proj_fun = proj_fun
        self.rate = rate
        self.Bias = np.array(Bias,dtype=mpf)
        self.Quaternion = np.array(quat,dtype=mpf)
        
        self.Quaternion[0:4] = self.Quaternion[0:4]/mp.norm(self.Quaternion)
        self.acc = np.zeros(3,dtype=mpf)
        self.speed = np.zeros(3,dtype=mpf)
        self.position = np.zeros(3,dtype=mpf)
        self.relative_step = np.zeros(3,dtype=mpf)
        self.Q = np.array(Q).reshape(2,2)
        self.R = np.array(R).reshape(3,3)
        self.Pk = np.array(Pk).reshape(6,6)
        self.time = time
        self.rotsurf = np.array(rotsurf,dtype=mpf)
        self.base_width=base_width
        self.state = np.zeros(6,dtype=mpf)
        self.state[3:6] = log_q(np.array(quat_mult(quat_inv(self.rotsurf),(self.Quaternion))))
        #self.state[3:6] = log_q(np.array(self.Quaternion))
        self.quat = np.array(quat_mult(quat_inv(self.rotsurf),quat_inv(self.Quaternion)),dtype=mpf)[[0,3]]
        self.normal = np.array(normal,dtype=mpf)
        self.center= np.zeros(3,dtype=mpf)
        self.surf_center= np.zeros(3,dtype=mpf)
        self.gravity = np.array(grav,dtype=mpf)
        self.gravity_r = np.array(quat_rot([0,*(grav)],quat_inv(self.Quaternion)))[1:4]
        self.mag0 = mag0
        self.pleft = mpf(0)
        self.pright = mpf(0)
        self.variant_update = None
        self.detection=detection
        
        
    def predict_speed(self,acc):
        self.speed = self.speed +acc*self.dt
    def predict_sspeed(self,acc):
        self.s_speed = self.s_speed +acc*self.dt
    def predict_position(self):
        self.position = self.position +self.speed*self.dt
    def predict_position_acc(self,acc):
        self.position = self.position +acc*(self.dt)**2
    def predict(self,leftw,rightw,Surface):
        
        self.state,self.Pk = predict_tested(self.state, self.Pk, self.Q, leftw,rightw, Surface,self.rotsurf,dw=self.base_width/2,dt=self.dt)
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

    def UpdateSensor(self, *args):
        Time, Surface, Accelerometer,Pressure = args[:4]
        self.dt = mpf(Time) - self.time
        self.time = mpf(Time)
        pstate = np.copy(self.state)

        normal = Surface[1:4]
        normal = normal/mp.norm(normal)
        self.normal = normal
        
        normal0 = np.array([0,0,1],dtype=mpf)
        self.rotsurf = quat_ntom(normal0,normal)
        Surface[0] =  np.dot(self.position,normal)
        Line = np.array([Pressure,0,0,1,*np.zeros(6)])
        Line0=  np.dot(self.position,[0,0,1])
        self.predict_speed(-self.gravity)
        self.predict_position()
        zero = Pressure
        if self.proj_fun == None:
            zero = Line0
        self.compute_center(np.array([Pressure,0,0,1,*np.zeros(6)]))
        
        
        quat=self.rotsurf
        self.Quaternion = normalize(quat_mult(quat,ExpQua(self.state[3:6])))
        n_pos = np.array(quat_rot([0,*(self.state[0:3]-pstate[0:3])],(quat)))[1:4]/self.dt**2
        step = np.array(n_pos,dtype=mpf)-self.speed/self.dt
        relative_step = np.array(quat_rot([0,*step], quat_inv(self.Quaternion)))[1:4]
        
        #self.relative_step = relative_step
        self.relative_step = Accelerometer
        
        if self.variant_update !=None:
            self.variant_update(*args)
        
        quat=self.rotsurf
        self.Quaternion = normalize(quat_mult(quat,ExpQua(self.state[3:6])))
        #self.Quaternion = ExpQua(self.state[3:6])#quat_mult(quat,ExpQua(self.state[3:6]))
        
        
        dx = np.array(quat_rot([0,1,0,0],self.Quaternion)[1:4],dtype=mpf)
        dy = np.cross(normal, dx)
        dy=dx
        
        n_pos = np.array(quat_rot([0,*(self.state[0:3]-pstate[0:3])],(quat)))[1:4]/self.dt**2
        #n_pos = np.array([*(self.state[0:3]-pstate[0:3])])[0:3]/self.dt**2
        step = np.array(n_pos,dtype=mpf)-self.speed/self.dt
        relative_step = np.array(quat_rot([0,*step], quat_inv(self.Quaternion)))[1:4]
        
        ax,ay,az = Accelerometer
        if self.proj_fun == None:
            acc_earth = step
            #acc_earth = np.array(quat_rot([0,ax,ay,az],self.Quaternion)[1:4],dtype=mpf)
        else:
            relative_step = np.array(quat_rot([0,*step], quat_inv(self.Quaternion)))[1:4]
            self.compute_center(np.array([Pressure,0,0,1,*np.zeros(6)]))
            self.Quaternion,self.Pk,acc_earth = self.proj_fun(self.dt, self.Quaternion, self.Pk, self.position, self.center, relative_step, Line,Surface)
        self.predict_position_acc(acc_earth)
        self.predict_speed(acc_earth)
