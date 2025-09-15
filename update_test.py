
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
from mpmath import matrix
from kfilterdata1 import KFilterDataSurf as KFilterData 
from predict_test import predict
from function_quat import *
from numpy.linalg import eig
from compute_state import compute_state_cross_product as compute_state
from compute_state import compute_state_cross_product_acconly as compute_state_acconly

N=100


def update0(Quaternion,Bias,Pk,RR,Gyroscope, Accelerometer,Magnetometer,Orient,mag0=np.array([0,1,0],dtype=mpf)):
    q = Quaternion
    qm=q
    rotation_matrix = (QuatToRot(np.array(quat_inv(Quaternion))))
    
    [ax,ay,az] = Accelerometer
    acc= Accelerometer
    acc0 = np.array([0,0,1],dtype=mpf)
    
    [mx,my,mz] = Magnetometer
    mag = Magnetometer
    g=1
    
    state,K,cov_correct=compute_state(acc,acc0,mag,mag0,rotation_matrix,Pk,RR)
    dv = state[0:3]
    
    
    Bias += np.array(state[3:6])
    Quaternion = np.array(quat_mult(Quaternion,ExpQua(dv)))
    Pk = Pk-cov_correct
    Pk = (Pk+Pk.T)/2
    Quaternion = c_normalize(Quaternion,qm)
    """
    Reset correction
    """
    G = np.eye(6).astype(mpf)
    G[0:3,0:3] = np.eye(3,dtype=mpf)-skewSymmetric(0.5*state[:3])
    Pk = np.dot(np.dot(G,Pk),G.T)
    return Quaternion,Bias,Pk    


def update_acconly(Quaternion,Bias,Pk,RR,Gyroscope, Accelerometer,Orient):
    q = Quaternion
    qm=q
    rotation_matrix = (QuatToRot(np.array(quat_inv(Quaternion))))
    
    [ax,ay,az] = Accelerometer
    acc= Accelerometer
    acc0 = np.array([0,0,1],dtype=mpf)

    state,K,cov_correct=compute_state_acconly(acc,acc0,rotation_matrix,Pk,RR)
    dv = state[0:3]
    
    Bias += np.array(state[3:6])
    Quaternion = np.array(quat_mult(Quaternion,ExpQua(dv)))
    Pk = Pk-cov_correct
    #print(Pk.astype(float),cov_correct.astype(float))
    Pk = (Pk+Pk.T)/2
    Quaternion = c_normalize(Quaternion,qm)
    """
    Reset correction
    """
    G = np.eye(6).astype(mpf)
    G[0:3,0:3] = np.eye(3,dtype=mpf)-skewSymmetric(0.5*state[:3])
    Pk = np.dot(np.dot(G,Pk),G.T)
    return Quaternion,Bias,Pk    

newset = KFilterData(N,mpf(1.),g_bias= 10**(-5),g_noise=10**(-6),a_noise=10**(-3)) 
newset = KFilterData(N,mpf(1.),g_bias= 10**(-5),g_noise=10**(-10),a_noise=10**(-5))#,g_bias= 10**(-2),g_noise=10**(-4))#,a_noise=10**(-3)) 

newset = KFilterData(N,mpf(1.))


def test_update(newset,qq = 10**(-5),rr=mpf(10**(-10))):
    N = len(newset.orient)
    compare = np.zeros((N,4))
    newcompare = np.zeros((N,4))
    bias = np.zeros((N,3))
    start = newset.orient[0,:]
    Pk = np.eye(6,dtype=mpf)
    Quat = start
    Bias = np.zeros(3,dtype=mpf)
    QQ = np.eye(6,dtype=mpf)
    QQ[:3,:3]*=qq
    QQ[3:6,3:6]*=10**(-5)
    RR = np.eye(6,dtype=mpf)*rr
    dt = newset.DT


    new_orient = newset.neworient
    for i in range(1,100):
        firstquat=Quat
        Quat,Bias,Cov,Phi = predict(Quat,Bias,Pk,QQ,newset.gyro[i,:],newset.orient[i,:],dt=dt)
        oldquat=Quat
        Quat,Bias,Cov = update0(Quat, Bias, Cov, RR, newset.gyro[i,:], newset.acc[i,:], newset.mag[i,:], newset.orient[i,:])

        Pk=Cov
        compare[i,:] = normalize(quat_mult(Quat,quat_inv(newset.orient[i,:])))-np.array([1,0,0,0])
        newcompare[i,:] = normalize(quat_mult(Quat,quat_inv(new_orient[i,:])))-np.array([1,0,0,0])
    return compare,newcompare,bias
        
    

compare,newcompare, bias = test_update(newset)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(bias)
ax.set_title("bias")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.diff(bias,axis=0))
ax.set_title("dbias")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.linalg.norm(compare.astype(float),axis=1)[10:])
ax.plot(np.linalg.norm(newcompare.astype(float),axis=1)[10:])
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([(compare[i,:]) for i in range(len(compare))],'*')

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([mp.norm(compare[i,:]) for i in range(len(compare))],'*')
ax.set_title('norm')
