
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
from kfilterdata_synthetic import KFilterDataSynth as KFilterData

mp.dps=40
def predict(orientation,Pk,QQ,leftw,rightw,Surface,quat,dt=mpf(1.),dw=mpf(1.),radius=mpf(1.),nticks=mpf(10000),coef=5164):
    normal1 = Surface[1:4]
    normal = np.array([0,0,1],dtype=mpf)
    rotsurf = quat_ntom(normal,normal1)

    dl = leftw#*mpf(2*mp.pi*radius)/nticks
    dr = rightw#*mpf(2*mp.pi*radius)/nticks

    d = mpf(dl+dr)/mpf(2)
    dtheta = mpf(dr-dl)/mpf(2*dw)
    theta = orientation[5]
    orientation[0:3] =orientation[0:3] +np.array([d*mp.cos(theta+dtheta/2),d*mp.sin(theta+dtheta/2),mpf(0)])#np.array(quat_rot(np.array([mpf(0),d*mp.cos(theta+dtheta/mpf(2)),d*mp.sin(theta+dtheta/mpf(2)),mpf(0)],dtype=mpf), quat),dtype=mpf)[1:4]
    orientation[3:6] = orientation[3:6]+np.array([0,0,mpf(dtheta)])
    
    
    
    
    Jx= np.eye(6)
    Ju = np.zeros((2,6))
    Jx[5,:2] = np.array([-d*mp.sin(theta),d*mp.cos(theta)],dtype=mpf)
    Ju[0,:2] = np.array([mp.cos(theta),mp.sin(theta)],dtype=mpf)
    Ju[1,:] = np.array([-0.5*d*mp.sin(theta),0.5*d*mp.cos(theta),0,0,0,1],dtype=mpf)
    Pk = Jx.T @ Pk @ Jx +Ju.T@QQ@Ju#QQk
    Pk = (Pk+Pk.T)/2
    return orientation,Pk
    
N=100    
newset = KFilterData(N,mpf(1.)/mpf(1.),mode='OdoAccPre',traj='Rand',lw_noise=0.1*0,rw_noise=0.1*0,g_bias= 10**(-4)*0,g_noise=10**(-10)*0,a_noise=10**(-2)*0,surf=np.array([0,1,-1],dtype=mpf)) 
def test_predict(newset,qq = 10**(-10)):
    N = len(newset.orient)
    compare = np.zeros((N,4))
    start = newset.orient[0,:]
    
    Pk = np.eye(6)
    Quat = start/mp.norm(start)
    Bias = np.zeros(3).astype(mpf)
    Bias = np.array([mpf(0),mpf(0),mpf(0)])
    Q = np.eye(2)*qq
    dt = newset.DT
    time = mpf(0)
    orientation = np.zeros(6,dtype=mpf)
    
    Surface=np.zeros(9)
    Surface[1:4]=newset.normal
    quat = newset.rotsurf
    orientation[:3] = newset.pos_earth[0,:]
    orientation[3:6] = log_q(np.array(quat_mult(quat_inv(quat),(start))))
    compare[0,:]=Quat

    for i in range(1,100):
        dt = mpf(i)/newset.freq - time
        time = mpf(i)/newset.freq
        orientation,Cov = predict(orientation,Pk,Q,newset.leftw[i]*dt,newset.rightw[i]*dt,Surface,quat,dt=dt,dw=newset.dw,radius=newset.radius,nticks=newset.nticks)
        
        Pk=Cov
        Quat = quat_mult(quat,ExpQua(orientation[3:6]))
        
        r_orientation = orientation[3:6]
        log_Quaternion = log_q(np.array(quat_mult(quat_inv(quat),np.array((newset.orient[i,:])))))
        
        compare[i,:]=Quat
        
    return compare
        
    


compare = test_predict(newset)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare)

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([mp.norm(compare[i,:]) for i in range(len(compare))],'*')
ax.set_title('norm')