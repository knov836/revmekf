
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
import sympy as sym
from sympy import symbols
from sympy import Rational
from sympy import Abs
from mpmath import mpf
from mpmath import mp
from mpmath import matrix


def SymExpRot(v):
    normv = sym.simplify(sym.sqrt(v.dot(v)))
    if normv==0:
        return sym.Matrix(np.eye(3))
    vv = sym.Matrix(v/normv).reshape(1,3)
    s = sym.Matrix([[Rational(0), -vv[2], vv[1]],
                [vv[2], Rational(0), -vv[0]],
                [-vv[1], vv[0], Rational(0)]])
    RotM = sym.eye(3) + sym.sin(normv)*s + (1-sym.cos(normv))*(s@s)
    return RotM


def SymExpRot2(v,normv):
    if normv==0:
        return sym.Matrix(np.eye(3))
    vv = sym.Matrix(v).reshape(1,3)
    s = sym.Matrix([[Rational(0), -vv[2], vv[1]],
                [vv[2], Rational(0), -vv[0]],
                [-vv[1], vv[0], Rational(0)]])
    RotM = sym.eye(3) + sym.sin(normv)*s + (Rational(1)-sym.cos(normv))*(s@s)
    return RotM

def SymExpRot3(v,c_normv):
    if c_normv==1:
        return sym.Matrix(np.eye(3))
    vv = sym.Matrix(v).reshape(1,3)
    s = sym.Matrix([[Rational(0), -vv[2], vv[1]],
                [vv[2], Rational(0), -vv[0]],
                [-vv[1], vv[0], Rational(0)]])
    RotM = sym.eye(3) + sym.sqrt(Abs(1-c_normv**2))*s + (Rational(1)-c_normv)*(s@s)
    
    return RotM
def project(p, f,gradf, tol=1e-40):
    u = p
    for i in range(100):
        g = gradf(u.astype(float))
        v = u - f(u) * g / np.linalg.norm(g)**2
        if np.linalg.norm(u - v) < tol:
            return v
        else:
            u = v
    return v
def skewSymmetric(v,dtype=mpf):
    return np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]],dtype=dtype) 

def RotSurf(normal):
    normal1 = np.array([0,0,1],dtype=mpf)
    Quaternion = quat_ntom(normal1,normal)
    return Quaternion
def quat_ntom(n,m):
    normal1 = np.array(m,dtype=mpf)
    normal = np.array(n,dtype=mpf)
    
    n0,n1 =mp.norm(normal),mp.norm(normal1)
    newquat = skewSymmetric(normal/n0)@normal1/(n1)
    sproduct = np.dot(normal/n0,normal1/n1)
    if np.abs(sproduct)>=1.0:
        beta = mpf(0)
    else:
        beta = mp.acos(sproduct)
    if beta!=0:
        newquat = newquat/mp.sin(beta)*mp.sin(beta/2)
    else:
        if mp.norm(normal/n0 +normal1/n1) == 0 and mp.norm(normal) !=0:
            beta = mp.pi
            newquat = np.array([0,0,1],dtype=mpf)
        else:
            newquat =newquat*mpf(0)
    Quaternion = np.zeros(4,dtype=mpf)
    Quaternion[1:4] = newquat
    Quaternion[0] = mp.cos(beta/2)
    return Quaternion


def quat_ntom2(n,m):
    normal1 = np.array(m,dtype=mpf)
    normal = np.array(n,dtype=mpf)
    
    n0,n1 =mp.norm(normal),mp.norm(normal1)
    newquat = skewSymmetric(normal/n0)@normal1/(n1)
    sproduct = np.dot(normal/n0,normal1/n1)
    beta = mp.acos(sproduct)
    if beta!=0:
        newquat = newquat/mp.sin(beta)*mp.sin(beta/2)
    else:
        if mp.norm(normal/n0 +normal1/n1) == 0 and mp.norm(normal) !=0:
            beta = mp.pi
            newquat = np.array([0,0,1],dtype=mpf)
        else:
            newquat =newquat*mpf(0)
    Quaternion = np.zeros(4,dtype=mpf)
    Quaternion[1:4] = newquat
    Quaternion[0] = mp.cos(beta/2)
    return Quaternion
def quat_inv(q):
    return [q[0],-q[1],-q[2],-q[3]]
def quat_mult(q0,q1):
    w0,x0,y0,z0 = q0
    w1,x1,y1,z1 = q1
    return [w0*w1-x0*x1-y0*y1-z0*z1, w0*x1+x0*w1+y0*z1-z0*y1,w0*y1+y0*w1+z0*x1-x0*z1,w0*z1+z0*w1+x0*y1-y0*x1]
def quat_rot(q0,q1):
    return quat_mult(q1,quat_mult(q0,quat_inv(q1)))


def quaternConj(q) :
    qConj = np.array( [-q[0], q[1], q[2], q[3]])
    
    return qConj

def normalize(q):
    qw,qx,qy,qz = q
    if mp.norm(qw)>10e-3:
        if qw<0:
            qw=-qw
            qx=-qx
            qy=-qy
            qz=-qz
        return np.array([qw,qx,qy,qz],dtype=mpf)
    i = np.argmax(np.abs(q))
    if q[i]<0:
        qw=-qw
        qx=-qx
        qy=-qy
        qz=-qz
    return np.array([qw,qx,qy,qz],dtype=mpf)

def c_normalize(q,qm):
    qw,qx,qy,qz = q
    
    if np.linalg.norm(q+qm)<np.linalg.norm(q-qm):
        qw=-qw
        qx=-qx
        qy=-qy
        qz=-qz
    return np.array([qw,qx,qy,qz],dtype=mpf)


def RotToQuat(M):
    tr = M[0,0]+M[1,1]+M[2,2]
    if tr>0:
        S = mp.sqrt(1+tr)*2
        qw = S/4
        qx = mpf(M[2,1]-M[1,2])/(S)
        qy = mpf(M[0,2]-M[2,0])/(S)
        qz = mpf(M[1,0]-M[0,1])/(S)
    elif M[0,0]>M[1,1] and M[0,0]>M[2,2]:
        S = mp.sqrt(1+M[0,0]-M[1,1]-M[2,2])*2
        qw = mpf(M[2,1]-M[1,2])/(S)
        qx = S/4
        qy = mpf(M[0,1]+M[1,0])/(S)
        qz = mpf(M[0,2]+M[2,0])/(S)
    elif M[1,1]>M[2,2]:
        S = mp.sqrt(1+M[1,1]-M[0,0]-M[2,2])*2
        qw = mpf(M[0,2]-M[2,0])/(S)
        qx = mpf(M[0,1]+M[1,0])/(S)
        qy = S/4
        qz = mpf(M[1,2]+M[2,1])/(S)
    else:
        S = mp.sqrt(1+M[2,2]-M[0,0]-M[1,1])*2
        qw = mpf(M[1,0]-M[0,1])/(S)
        qx = mpf(M[0,2]+M[2,0])/(S)
        qy = mpf(M[1,2]+M[2,1])/(S)
        qz = S/4
    if qw<0:
        qw=-qw
        qx=-qx
        qy=-qy
        qz=-qz
    return np.array([qw,qx,qy,qz],dtype=mpf)

def ExpQua(v):
    q = np.zeros([4], dtype=mpf)
    v = v / mpf(2)
    theta = mp.norm(v)
    if theta != 0:
        q[0] = mp.cos(theta)
        q[1:] = v/theta*mp.sin(theta)
    else:
        q[0] = mpf(1)
    return q

def sym_ExpQua(v):
    q = [0,0,0,0]
    v = v / 2
    theta = sym.sqrt(np.array(v).dot(np.array(v)))
    if theta != 0:
        q[0] = sym.cos(theta)
        q[1:] = v/theta*sym.sin(theta)
    else:
        q[0] = Rational(1)
    return q

def log_q(q):
    if mp.norm(q[1:]) == 0:
        return np.zeros(3,dtype=mpf)
    phi = mpf(2)*mp.atan2(mp.norm(q[1:]),q[0])
    phi = mpf(2)*mp.acos(q[0])
    if phi<-mp.pi:
        phi+=2*mp.pi
    if phi>mp.pi:
        phi-=2*mp.pi
            
    
    u = q[1:]/mp.norm(q[1:])
    return u*phi


def sym_log_q(q):
    if sym.Matrix(q[1:]).dot(sym.Matrix(q[1:])) == 0:
        return [0,0,0]
    #phi = Rational(2)*sym.atan(sym.srqt(q[1:].dot(q[1:]))/q[0])
    phi = Rational(2)*sym.acos(q[0])
    
    u = sym.Matrix(q[1:])/sym.sqrt(sym.Matrix(q[1:]).dot(sym.Matrix(q[1:])))
    return u*phi
    
def log_R(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    phi = mp.acos((tr-1)/2)
    W = (R-R.T)
    ihat = np.array([W[2,1],W[0,2],W[1,0]])
    u = ihat/(2*mp.sin(phi))
    return u*phi
def ExpRot(v):
    normv = mp.norm(v)
    if normv==0:
        return np.eye(3,dtype=mpf)
    vv = np.array(v)/normv
    s = np.array([[0, -vv[2], vv[1]],
                [vv[2], 0, -vv[0]],
                [-vv[1], vv[0], 0]], dtype=mpf)
    vv=vv[:,np.newaxis]
    RotM = np.identity(3,dtype=mpf) + mp.sin(normv)*s + (1-mp.cos(normv))*(s@s)
    
    return RotM

def angle_between(v1, v2):
    #print(v1,v2)
    nv1 = mp.norm(v1)
    nv2 = mp.norm(v2)
    v1_u = np.array(v1/nv1).astype(mpf)
    v2_u = np.array(v2/nv2).astype(mpf)
    a,b,c = v1
    e,f,g = v2
    inte = 0.5*(np.dot(v1_u,v1_u)+np.dot(v2_u,v2_u)-np.dot(v1_u-v2_u,v1_u-v2_u))
    cc = np.dot(skewSymmetric(v1_u),v2_u)

    return mp.asin(np.clip(mp.norm(cc), -1.0, 1.0))#*np.random.choice([-1,1])

def Quaternion2RotationMatrix(q):
    q_scalar = q[0]
    q_vector = q[1:4][:, np.newaxis]
    q_skew = np.array([[0, -q[3], q[2]], 
                       [q[3], 0, -q[1]], 
                       [-q[2], q[1], 0]],dtype=mpf)
    rotation_matrix = (q_scalar**2 - np.linalg.norm(q_vector)**2)*np.identity(3) + 2 * q_scalar * q_skew + 2 * q_vector @ q_vector.T

    return rotation_matrix

def QuatToRot(q):
    M = np.zeros((3,3),dtype=mpf)
    qw,qx,qy,qz = q
    M[0,0] = 1-2*qy**2-2*qz**2
    M[0,1] = 2*qx*qy-2*qz*qw
    M[0,2] = 2*qx*qz+2*qy*qw
    M[1,0] = 2*qx*qy+2*qz*qw
    M[1,1] = 1-2*qx**2-2*qz**2
    M[1,2] = 2*qy*qz-2*qx*qw
    M[2,0] = 2*qx*qz-2*qy*qw
    M[2,1] = 2*qy*qz+2*qx*qw
    M[2,2] = 1-2*qx**2-2*qy**2
    return M



a = np.random.random_sample(3)*np.pi
u = ExpRot(a)


q = ExpQua(a)


u = np.array([1,0,0])
v= np.array([0,1,1])

qu = ExpQua(u)
qv = ExpQua(v)
ru = ExpRot(u)
rv = ExpRot(v)