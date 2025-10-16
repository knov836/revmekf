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
from numpy.linalg import eig


def h_mat_acc(rotation_matrix,acc0):
    H = np.zeros((3,6),dtype=mpf)
    old_acc_rot = np.array(rotation_matrix @ acc0).reshape((3,1))
    old_acc_rot = old_acc_rot/mp.norm(old_acc_rot.flatten())
    
    H[0:3,3:6] = np.eye(3) - old_acc_rot@old_acc_rot.T
    return H

def h_mat(rotation_matrix,acc0,mag0):
    H = np.zeros((6,6),dtype=mpf)
    old_acc_rot = np.array(rotation_matrix @ acc0).reshape((3,1))
    old_acc_rot = old_acc_rot/mp.norm(old_acc_rot.flatten())
    old_mag_rot = np.array(rotation_matrix @ mag0).reshape((3,1))
    old_mag_rot = old_mag_rot/mp.norm(old_mag_rot.flatten())
    H[0:3,0:3] = np.eye(3) - old_acc_rot@old_acc_rot.T
    H[3:6,0:3] = np.eye(3) - old_mag_rot@old_mag_rot.T
    return H


def h_mat_add(rotation_matrix,acc0,mag0):
    H = np.zeros((6,6),dtype=mpf)
    old_acc_rot = np.array(rotation_matrix @ acc0).reshape((3))
    old_acc_rot = old_acc_rot/mp.norm(old_acc_rot.flatten())
    old_mag_rot = np.array(rotation_matrix @ mag0).reshape((3))
    old_mag_rot = old_mag_rot/mp.norm(old_mag_rot.flatten())
    H[0:3,0:3] = skewSymmetric(old_acc_rot)
    H[3:6,0:3] = skewSymmetric(old_mag_rot)
    return H

def kalman_gain(Pk,H,RR):
    S=matrix(H @ Pk @ H.T + RR)
    iS = np.array(S**(-1),dtype=mpf).reshape(RR.shape)
    S = np.array(S,dtype=mpf).reshape(RR.shape)
    K= Pk @ H.T @ iS
    return K,K@S@K.T

def compute_state(acc,acc0,mag,mag0,rotation_matrix,Pk,RR,test=0):
    ax,ay,az=acc
    mx,my,mz=mag
    v_m0 = np.array([ax,ay,az],dtype=mpf)
    v_t0 = (rotation_matrix) @ acc0
    
    v_m1 = np.array([mx,my,mz],dtype=mpf)
    v_t1 = (rotation_matrix) @ mag0

    yy = np.concatenate((v_m0-v_t0,v_m1-v_t1))
    hh = h_mat_add(rotation_matrix,acc0,mag0)
    kk,cov_correct = kalman_gain(Pk,hh,RR)
    state = kk@yy.T

    return state,kk,cov_correct

def compute_state_cross_product_debug(acc,acc0,mag,mag0,rotation_matrix,Pk,RR,test=0):
    ax,ay,az=acc
    mx,my,mz=mag
    v_m0 = np.array([ax,ay,az],dtype=mpf)
    v_m0 = (v_m0/mp.norm(v_m0)).astype(mpf)
    v_t0 = (rotation_matrix) @ acc0
    
    v_m1 = np.array([mx,my,mz],dtype=mpf)
    v_m1 = (v_m1/mp.norm(v_m1)).astype(mpf)
    v_t1 = (rotation_matrix) @ mag0

    yy0 = np.dot(skewSymmetric(v_m0),v_t0).flatten()
    yy1 = np.dot(skewSymmetric(v_m1),v_t1).flatten()
    yy = np.concatenate((yy0,yy1))
    
    hh = h_mat(rotation_matrix,acc0,mag0)
    kk,cov_correct = kalman_gain(Pk,hh,RR)
    state = kk@yy.T
    print("state computed",state)
    
    print("debug")
    N = np.zeros((6,6),dtype=mpf)
    N[:3,:3] = skewSymmetric(v_m0)@skewSymmetric(v_m0)
    N[3:6,:3]= skewSymmetric(v_m1)@skewSymmetric(v_m1)
    print(np.dot(-N,state))
    M = np.zeros((6,6),dtype=mpf)
    M[:3,:3] = skewSymmetric(v_t0)@skewSymmetric(v_t0)
    M[3:6,:3]= skewSymmetric(v_t1)@skewSymmetric(v_t1)
    print(np.dot(-M,state))
    print(yy)
    print(hh+M)
    print("==================================")
    print(np.dot(-M,state))
    print("toto",v_t0,state[:3])
    print("end")
    print(np.dot(-skewSymmetric(v_t0)@skewSymmetric(v_t0),state[:3]))
    print(skewSymmetric(v_t0)@ExpRot(state[:3])@v_t0)
    print("compare")
    print(ExpRot(state[:3])@v_t0, skewSymmetric(v_t0)@state[:3],v_m0)
    print(ExpRot(state[:3])@v_t1, skewSymmetric(v_t1)@state[:3],v_m1)
    print(skewSymmetric(v_t0)@skewSymmetric(v_t0)@state[:3],skewSymmetric(v_t0)@v_m0)
    print(skewSymmetric(v_t1)@skewSymmetric(v_t1)@state[:3],skewSymmetric(v_t1)@v_m1)
    print("compare",(yy-hh@state),yy)
    print(rotation_matrix)
    print("end compare")
    
    return state,kk,cov_correct
def compute_state_cross_product(acc,acc0,mag,mag0,rotation_matrix,Pk,RR,test=0):
    ax,ay,az=acc
    mx,my,mz=mag
    v_m0 = np.array([ax,ay,az],dtype=mpf)
    v_m0 = (v_m0/mp.norm(v_m0)).astype(mpf)
    v_t0 = (rotation_matrix) @ acc0
    
    v_m1 = np.array([mx,my,mz],dtype=mpf)
    v_m1 = (v_m1/mp.norm(v_m1)).astype(mpf)
    v_t1 = (rotation_matrix) @ mag0

    yy0 = np.dot(skewSymmetric(v_m0),v_t0).flatten()
    yy1 = np.dot(skewSymmetric(v_m1),v_t1).flatten()
    yy = np.concatenate((yy0,yy1))

    hh = h_mat(rotation_matrix,acc0,mag0)
    kk,cov_correct = kalman_gain(Pk,hh,RR)
    state = kk@yy.T
    
    
    return state,kk,cov_correct


def compute_state_cross_product_acconly(acc,acc0,rotation_matrix,Pk,RR,test=0):
    ax,ay,az=acc
    v_m0 = np.array([ax,ay,az],dtype=mpf)
    v_m0 = (v_m0/mp.norm(v_m0)).astype(mpf)
    v_t0 = (rotation_matrix) @ acc0
    

    yy0 = np.dot(skewSymmetric(v_m0),v_t0).flatten()

    hh = h_mat_acc(rotation_matrix,acc0)
    kk,cov_correct = kalman_gain(Pk,hh,RR)
    state = kk@yy0.T
    
    return state,kk,cov_correct


def compute_state0(acc,acc0,mag,mag0,rotation_matrix,Pk,RR):
    ax,ay,az=acc
    mx,my,mz=mag
    v_m0 = np.array([ax,ay,az],dtype=mpf)
    v_m0 = v_m0/mp.norm(v_m0)
    v_t0 = (rotation_matrix) @ acc0
    
    v_m1 = np.array([mx,my,mz],dtype=mpf)
    v_m1 = v_m1/mp.norm(v_m1)
    v_t1 = (rotation_matrix) @ mag0

    yy0 = np.dot(skewSymmetric(v_m0),v_t0).flatten()
    yy1 = np.dot(skewSymmetric(v_m1),v_t1).flatten()
    yy = np.concatenate((yy0,yy1))
    
    hh = h_mat(rotation_matrix,acc0,mag0)
    kk,cov_correct = kalman_gain(Pk,hh,RR)
    state = kk@yy.T
    
    return state,kk,cov_correct
