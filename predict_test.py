
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
from mpmath import matrix
from function_quat import *
from kfilterdata1 import KFilterDataSurf as KFilterData

mp.dps = 40

def predict(Quaternion,Bias,Pk,QQ,Gyroscope,Orient,dt=1.):
    """
    Quaternion = R_B^A (from B to A (world frame))
    """
    Q = Quaternion
    [p,q,r] = np.array(Gyroscope)
    B = Bias
    
    wx = mpf(p-B[0])
    wy = mpf(q-B[1])
    wz = mpf(r-B[2])

    Quaternion = np.array(quat_mult(Q.T,ExpQua((np.array([wx,wy,wz]))*dt)))
    

    Phi = np.zeros((6,6),dtype=mpf)
    F = np.zeros((6,6),dtype=mpf)
    F[0:3,0:3] =  -skewSymmetric((np.array([wx,wy,wz])))
    F[0:3,3:6] = -np.eye(3)

    Phi= np.eye(6)+F*dt
    Psi= Phi
    fac = 1.
    for i in range(2,3):
        fac*=i
        Psi = Psi+((F)**i)*(dt**i)/fac
    QQk=  np.zeros((6,6))
    gyro_cov_mat = QQ[0,0]*np.eye(3)
    gyro_bias_cov_mat = QQ[3,3]*np.eye(3)
    QQk[0:3,0:3] = gyro_cov_mat*dt + gyro_bias_cov_mat*(dt**3)/3.0
    QQk[0:3,3:6] = -gyro_bias_cov_mat*(dt**2)/2.0
    QQk[3:6,0:3] = -gyro_bias_cov_mat*(dt**2)/2.0
    QQk[3:6,3:6] = gyro_bias_cov_mat*(dt**2)/2.0
    Pk = Phi @ Pk @ Phi.T +QQ#QQk
    Pk = (Pk+Pk.T)/2

    return Quaternion,B,Pk,Phi
    
N=100    
newset = KFilterData(N,mpf(1.),g_bias= 10**(-5),g_noise=10**(-6),a_noise=10**(-4)) 
angle = int(N/2)
newset = KFilterData(N,mpf(10.),alpha=angle) 

def test_predict(newset,qq = 10**(-10)):
    N = len(newset.orient)
    compare = np.zeros((N,4))
    start = newset.orient[0,:]
    Pk = np.eye(6)
    Quat = start/mp.norm(start)
    Bias = np.zeros(3).astype(mpf)
    Bias = np.array([mpf(0),mpf(0),mpf(0)])
    Q = np.eye(6)*qq
    dt = newset.DT
    #for i in range(1,N):
    time = mpf(0)
    for i in range(1,N):
        print("input",Quat,Bias)
        dt = mpf(i)/newset.freq - time
        time = mpf(i)/newset.freq
        Quat,Bias,Cov,Phi = predict(Quat,Bias,Pk,Q,newset.gyro[i,:],newset.orient[i,:],dt=dt)
        #print(Phi)
        Pk=Cov
        #compare[i,:] = Quat-newset.orient[i,:]
        compare[i,:] = normalize(quat_mult(Quat,quat_inv(newset.orient[i,:])))-np.array([1,0,0,0])
    return compare
        
    


compare = test_predict(newset)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare)

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([mp.norm(compare[i,:]) for i in range(len(compare))],'*')
#ax.plot([mp.norm(compare2[i,:]) for i in range(len(orient))],'*')
ax.set_title('norm')