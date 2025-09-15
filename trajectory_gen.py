
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numdifftools as nd 

from mpmath import mp
from mpmath import mpf
from function_quat import *

import sys,os



def project(p, f,gradf, tol=1e-10):
    u = p
    for i in range(100):
        g = gradf(u.astype(float))
        v = u - f(u) * g / np.linalg.norm(g)**2
        if np.linalg.norm(u - v) < tol:
            return v
        else:
            u = v
    return v

def position_data_odo(N,Surface,dt,dwheels,radius,nticks,delta):
    Surf = np.copy(Surface)
    Surface = Surface.astype(float)
    surf = lambda x : Surface[0] + Surface[1]*x[0]+ Surface[2]*x[1]+ Surface[3]*x[2]+ Surface[4]*x[0]*x[1]+ Surface[5]*x[1]*x[2]+ Surface[6]*x[2]*x[0] + Surface[7]*x[0]**2+ Surface[8]*x[1]**2+ Surface[9]*x[2]**2
    grad = nd.Gradient(surf) #
    
    
    normal1 = np.array([mpf(s) for s in Surf[1:4]],dtype=mpf)
    normal1 = normal1/mp.norm(normal1)
    normal = np.array([0,0,1],dtype=mpf)
    Quaternion = quat_ntom(normal,normal1)
    rot = QuatToRot(Quaternion)
    
    
    pos_earth = np.array(np.zeros((N,3)),dtype=mpf)
    
    
    dthetas = np.zeros(N,dtype=mpf)
    thetas = np.zeros(N,dtype=mpf)
    ddws = np.zeros(N,dtype=mpf)
    orient = np.zeros((N,4),dtype=mpf)
    orient[:,0] = mpf(1)
    
    mu, sigma = 0, np.pi*delta*(dt)
    quat = ExpQua(np.array([0,0,0],dtype=mpf))
    orient[0,:]=(Quaternion)
    for i in range(1,N):
        dtheta = np.random.normal(mu,sigma,size=1)
        dthetas[i] = mpf(dtheta[0])
        thetas[i] = mpf(dtheta[0])+thetas[i-1]
        dyaw_q = np.array(normal1*dtheta[0],dtype=mpf)
        dquat = ExpQua(dyaw_q)    
        quat = quat_mult(quat, dquat)
        qq = quat_inv(quat_mult(quat_inv(Quaternion),quat_inv(quat)))
        orient[i,:] = normalize(qq)
    
    
    j=1
    next_step = np.zeros(3,dtype=mpf)
    mu, sigma = 0, delta*(dt)**2
    while j<N:
        dw = np.random.normal(mu,sigma,size=1)
        ddws[j]=mpf(dw[0])
        next_step = np.array([dw[0]*mp.cos(thetas[j-1]+mpf(dthetas[j])/mpf(2)),dw[0]*mp.sin(thetas[j-1]+mpf(dthetas[j])/mpf(2)),mpf(0)],dtype=mpf)
        pos_earth[j,:] = pos_earth[j-1,:]+rot@next_step
        j=j+1
    leftw = np.zeros(N,dtype=mpf)
    rightw = np.zeros(N,dtype=mpf)
    for i in range(1,N):
        X = 2*ddws[i]
        Y = 2*dwheels*dthetas[i]
        
        dleft = mpf(X-Y)/mpf(2)
        dright = mpf(X+Y)/mpf(2)
        
        leftw[i] = (mpf(dleft*nticks/mpf(2*mp.pi*radius)))
        rightw[i] = (mpf(dright*nticks/mpf(2*mp.pi*radius)))
    return pos_earth,orient,Quaternion,leftw,rightw,dthetas,thetas,ddws

def position_data_rand_3D(N,delta):
    normal = np.array([0,0,1],dtype=mpf)
    
    
    pos_earth = np.array(np.zeros((N,3)),dtype=mpf)
    Quaternion = np.array([mpf(1),mpf(0),mpf(0),mpf(0)],dtype=mpf)
    rot = QuatToRot(Quaternion)
    
    #pos_earth[0,:] = np.copy(project(pos_earth[0,:], surf,grad))
    
    j=1
    next_step = np.zeros(3)
    mu, sigma = 0, delta
    while j<N:
        next_step[:] = np.random.normal(mu,sigma,size=3)
        pos_earth[j,:] = pos_earth[j-1,:]+rot@next_step
        j=j+1
    orient = np.zeros((N,4),dtype=mpf)
    orient[:,0] = 1
    
    mu, sigma = 0, np.pi/100
    quat = ExpQua(np.random.normal(mu,sigma,size=3))
    
    for i in range(N):
        dquat = ExpQua(np.random.normal(mu,sigma,size=3))    
        quat = quat_mult(quat, dquat)
        qq = quat_inv(quat_mult(quat_inv(Quaternion),quat_inv(quat)))
        orient[i,:] = normalize(qq)
    return pos_earth,orient,Quaternion


def position_data_rand(N,Surface,delta):
    Surf = np.copy(Surface)
    Surface = Surface.astype(float)
    surf = lambda x : Surface[0] + Surface[1]*x[0]+ Surface[2]*x[1]+ Surface[3]*x[2]+ Surface[4]*x[0]*x[1]+ Surface[5]*x[1]*x[2]+ Surface[6]*x[2]*x[0] + Surface[7]*x[0]**2+ Surface[8]*x[1]**2+ Surface[9]*x[2]**2
    grad = nd.Gradient(surf) #
    
    
    normal1 = np.array([mpf(s) for s in Surf[1:4]],dtype=mpf)
    normal = np.array([0,0,1],dtype=mpf)
    Quaternion = quat_ntom(normal,normal1)
    rot = QuatToRot(Quaternion)
    
    
    pos_earth = np.array(np.zeros((N,3)),dtype=mpf)
    
    
    pos_earth[0,:] = np.copy(project(pos_earth[0,:], surf,grad))
    
    j=1
    next_step = np.zeros(3)
    mu, sigma = 0, delta
    while j<N:
        next_step[:2] = np.random.normal(mu,sigma,size=2)
        pos_earth[j,:] = pos_earth[j-1,:]+rot@next_step
        j=j+1
    orient = np.zeros((N,4),dtype=mpf)
    orient[:,0] = 1
    
    mu, sigma = 0, np.pi/100
    quat = ExpQua(np.random.normal(mu,sigma,size=3))
    for i in range(N):
        dquat = ExpQua(np.random.normal(mu,sigma,size=3))    
        quat = quat_mult(quat, dquat)
        qq = quat_inv(quat_mult(quat_inv(Quaternion),quat_inv(quat)))
        orient[i,:] = normalize(qq)
    return pos_earth,orient,Quaternion


def position_data_rand_smallyaw(N,Surface):
    Surf = np.copy(Surface)
    Surface = Surface.astype(float)
    surf = lambda x : Surface[0] + Surface[1]*x[0]+ Surface[2]*x[1]+ Surface[3]*x[2]+ Surface[4]*x[0]*x[1]+ Surface[5]*x[1]*x[2]+ Surface[6]*x[2]*x[0] + Surface[7]*x[0]**2+ Surface[8]*x[1]**2+ Surface[9]*x[2]**2
    grad = nd.Gradient(surf) #
    
    
    normal1 = np.array([mpf(s) for s in Surf[1:4]],dtype=mpf)
    normal = np.array([0,0,1],dtype=mpf)
    Quaternion = quat_ntom(normal,normal1)
    rot = QuatToRot(Quaternion)
    
    
    pos_earth = np.array(np.zeros((N,3)),dtype=mpf)
    
    
    pos_earth[0,:] = np.copy(project(pos_earth[0,:], surf,grad))
    
    j=1
    next_step = np.zeros(3)
    mu, sigma = 0, 0.1
    while j<N:
        next_step[:2] = np.random.normal(mu,sigma,size=2)
        pos_earth[j,:] = pos_earth[j-1,:]+rot@next_step
        j=j+1
    orient = np.zeros((N,4),dtype=mpf)
    orient[:,0] = 1
    
    sigma = np.pi/100
    
    mu= np.array([0,0,1])
    cov = np.diag([sigma,sigma,sigma])
    
    alpha = np.random.normal(0,np.pi)
    #quat = ExpQua(np.random.normal(mu,sigma,size=3))
    quat = ExpQua(alpha*np.random.multivariate_normal(mu,cov))
    for i in range(N):
        alpha = np.random.uniform(-np.pi,np.pi)    
        nquat = ExpQua(alpha*np.random.multivariate_normal(mu,cov))
        log_dquat0= np.array(log_q(np.array(quat_mult(quat_inv(quat), nquat))))
        log_dquat0 = log_dquat0/np.linalg.norm(log_dquat0)*np.random.normal(0,sigma)
        
        quat = quat_mult(quat, ExpQua(log_dquat0))
        print(log_q(np.array(quat)))
        qq = quat_inv(quat_mult(quat_inv(Quaternion),quat_inv(quat)))
        orient[i,:] = normalize(qq)
    return pos_earth,orient,Quaternion

def test_traj0(N,alpha,Surface):
    pos_earth = np.array(np.zeros((N,3)),dtype=mpf)
    
    for i in range(5):
        pos_earth[i,0]=mpf(i)#/1000
        pos_earth[i,1]=mpf(i)#/1000
        pos_earth[i,2]=mpf(i)
    for i in range(5,10):
        pos_earth[i,0]=mpf(10-i)#/1000
        pos_earth[i,1]=mpf(10-i)#/1000
        pos_earth[i,2]=mpf(10-i)
    orient = np.zeros((N,4),dtype=mpf)
    angle = mpf(2)*mp.pi
    
    Surf = np.copy(Surface)
    Surface = Surface.astype(float)
    surf = lambda x : Surface[0] + Surface[1]*x[0]+ Surface[2]*x[1]+ Surface[3]*x[2]+ Surface[4]*x[0]*x[1]+ Surface[5]*x[1]*x[2]+ Surface[6]*x[2]*x[0] + Surface[7]*x[0]**2+ Surface[8]*x[1]**2+ Surface[9]*x[2]**2
    grad = nd.Gradient(surf) #
    
    normal1 = np.array([mpf(s) for s in Surf[1:4]],dtype=mpf)
    normal = np.array([0,0,1],dtype=mpf)    
    
    Quaternion = quat_ntom(normal,normal1)
    for i in range(N):
        if i<alpha:
            theta = mpf(i*angle)/mpf(N)
        else:
            theta = mpf((2*alpha-i)*angle)/mpf(N)
        dyaw_q = np.array(normal1*theta,dtype=mpf)
        yaw_q= ExpQua(dyaw_q)    
        
        position = project(pos_earth[i,:], surf,grad)
        pos_earth[i,:] = position
        qq = quat_inv(quat_mult(quat_inv(Quaternion),quat_inv(yaw_q)))
        orient[i,:] = normalize(qq)
    return pos_earth,orient,Quaternion
