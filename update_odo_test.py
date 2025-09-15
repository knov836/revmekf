
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
from mpmath import matrix
from kfilterdata_synthetic import KFilterDataSynth as KFilterData
from predict_odo_test import predict
from function_quat import *
from numpy.linalg import eig
from compute_state import compute_state_cross_product_acconly as compute_state

N=100



def update0(state,Pk,RR, Accelerometer,Acc0,Surface):
    
    normal1 = Surface[1:4]
    normal = np.array([0,0,1],dtype=mpf)
    rotsurf = quat_ntom(normal,normal1)
    
    
    Quaternion = ExpQua(state[3:6])
    q = Quaternion
    qm=q
    rotation_matrix = (QuatToRot(np.array(quat_inv(Quaternion))))
    
    [ax,ay,az] = Accelerometer
    acc= Accelerometer
    acc0 = np.array(Acc0,dtype=mpf)

    state_correct,K,cov_correct=compute_state(acc,acc0,rotation_matrix,Pk,RR)
    A_proj = np.zeros((5,6))
    A_proj[:,:5]=np.eye(5)
    Gamma = A_proj.T@np.linalg.inv(A_proj@A_proj.T)
    state_correct = state_correct-Gamma@(A_proj@state_correct-np.zeros(5,dtype=mpf))
    
    
    dv = state_correct[3:6]
    Quaternion = np.array(quat_mult(Quaternion,ExpQua(dv)))
    state[5]+=state_correct[5]
    Pk = Pk-cov_correct
    Pk = (np.eye(6)-Gamma@A_proj)@Pk
    Pk = (Pk+Pk.T)/2
    Quaternion = c_normalize(Quaternion,qm)
    """
    Reset correction
    """
    G = np.eye(6).astype(mpf)
    G[3:6,3:6] = np.eye(3,dtype=mpf)-skewSymmetric(0.5*state[3:6])
    Pk = np.dot(np.dot(G,Pk),G.T)
    return state,Pk    


newset = KFilterData(N,mpf(1.)/mpf(1.),mode='OdoAccPre',traj='Rand',lw_noise=0.1*0,rw_noise=0.1*0,g_bias= 10**(-4)*0,g_noise=10**(-10)*0,a_noise=10**(-2)*0,surf=np.array([0,1,-1],dtype=mpf)) 


def test_update(newset,qq = 10**(-5),rr=mpf(10**(-10))):
    N = len(newset.orient)
    compare = np.zeros((N,4))
    newcompare = np.zeros((N,4))
    
    start = newset.orient[0,:]
    Pk = np.eye(6,dtype=mpf)
    Quat = start

    QQ = np.eye(2,dtype=mpf)
    QQ*=qq
    RR = np.eye(3,dtype=mpf)*rr
    dt = newset.DT
    time = mpf(0)

    new_orient = newset.neworient
    
    Surface=np.zeros(9)
    Surface[1:4]=newset.normal
    normal = newset.normal
    normal = normal/mp.norm(normal)
    quat = newset.rotsurf
    
    
    orientation = np.zeros(6,dtype=mpf)
    orientation[:3] = newset.pos_earth[0,:]
    orientation[3:6] = log_q(np.array(quat_rot(quat_mult((start),quat_inv(quat)),quat_inv(quat))))
    for ind in range(1,99):
        dt = mpf(ind)/newset.freq - time
        time = mpf(ind)/newset.freq
        
        firstquat=Quat
        orientation,Cov = predict(orientation,Pk,QQ,newset.leftw[ind],newset.rightw[ind],Surface,quat,dt=dt,dw=newset.dw,radius=newset.radius,nticks=newset.nticks)
        oldquat=orientation
        r_orientation = normal*orientation[5]
        qq_orientation = ExpQua(orientation[3:6])
        q_orientation = ExpQua(r_orientation)
        transformed_acc = np.array(quat_rot([0,*newset.acc[ind,:]],quat_inv(quat)))[1:4]
        transformed_acc2 = np.array(quat_rot([0,0,0,-1],quat_mult(quat,q_orientation)))[1:4]
        transformed_acc3 = np.array(quat_rot([0,0,0,-1],q_orientation))[1:4]
        transformed_acc2 = np.array(quat_rot(quat_rot([0,0,0,-1],quat_inv(quat)),qq_orientation))[1:4]
        transformed_acc4 = np.array(quat_rot(quat_rot([0,0,0,-1],quat_inv(quat)),quat_mult(quat,qq_orientation)))[1:4]
        transformed_acc0 = np.array(quat_rot([0,0,0,1],(quat)))[1:4]
        transformed_acc1 = np.array(quat_rot([0,0,0,1],(q_orientation)))[1:4]
        acc = np.array(quat_rot(quat_rot([0,*newset.gravs[ind,:]],quat_inv(quat)),quat_inv(quat)))[1:4]
        acc0 = np.array(quat_rot([0,0,0,1],quat_inv(quat)))[1:4]
        acc = newset.acc[ind,:]
        orientation,Cov = update0(orientation, Cov, RR, acc,acc0 , Surface)
        
        Pk=Cov
        compare[ind,:] = normalize(quat_mult(quat_mult(quat,ExpQua(orientation[3:6])),quat_inv(newset.orient[ind,:])))-np.array([1,0,0,0])
        newcompare[ind,:] = normalize(quat_mult(quat_mult(quat,ExpQua(orientation[3:6])),quat_inv(new_orient[ind,:])))-np.array([1,0,0,0])
        acc = np.array(quat_rot(quat_rot([0,*newset.gravs[ind,:]],quat_inv(quat)),quat_inv(quat)))[1:4]
        acc0 = np.array(quat_rot([0,0,0,1],quat_inv(quat)))[1:4]
        acc = newset.acc[ind,:]
        qq_orientation = ExpQua(orientation[3:6])
        r_orientation = orientation[3:6]
        log_Quaternion = log_q(np.array(quat_mult(quat_inv(quat),np.array((newset.orient[ind,:])))))
    return compare,newcompare
        
compare,newcompare= test_update(newset)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.linalg.norm(compare.astype(float),axis=1))
ax.set_title("compare")

