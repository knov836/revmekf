
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

def quat_to_pos(time,quaternion,acc,gravity):
    N = len(time)
    time0 = time-time[0]
    
    speed = np.zeros((N,3))
    pos = np.zeros((N,3))
    for i in range(1,N):
        dt =(time0[i]-time0[i-1])
        acc_earth = np.array(quat_rot([0,*(acc[i,:])],quaternion[i,:]))[1:4]
        print(acc_earth)
        speed[i,:]=speed[i-1,:]+(acc_earth-gravity)*(dt)
        pos[i,:] = pos[i-1,:]+speed[i,:]*dt
    return pos,speed



def acc_mag_to_rotation(acc, mag, declination_deg=0.0, degrees=False):
    """
    Compute rotation (roll, pitch, yaw) and return rotation matrix and quaternion
    from accelerometer and magnetometer measurements.

    Inputs:
      acc: (3,) array-like - accelerometer reading in body frame (Ax, Ay, Az)
      mag: (3,) array-like - magnetometer reading in body frame (Mx, My, Mz)
      declination_deg: optional magnetic declination to add to heading (degrees)
      degrees: if True return Euler angles in degrees, else radians

    Returns:
      dict with keys:
        'roll','pitch','yaw' : Euler angles (radians by default)
        'R' : 3x3 rotation matrix that maps body -> world (ENU: x=E, y=N, z=Up)
        'quat' : quaternion in [x, y, z, w] format (same frame mapping)
    """
    acc = np.asarray(acc, dtype=float)
    mag = np.asarray(mag, dtype=float)

    # normalize
    if np.linalg.norm(acc) == 0 or np.linalg.norm(mag) == 0:
        raise ValueError("acc or mag vector has zero length")

    ax, ay, az = acc / np.linalg.norm(acc)
    mx, my, mz = mag / np.linalg.norm(mag)

    # Compute roll and pitch from accelerometer (body frame)
    # roll: rotation about x, pitch: rotation about y
    roll  = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az))

    # Tilt compensate magnetometer
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)

    # using common tilt-compensation formulas
    Mx_comp = mx * cp + mz * sp
    My_comp = mx * sr * sp + my * cr - mz * sr * cp

    # heading (yaw). sign conventions vary; this returns yaw = 0 when facing +X_body -> +X_world
    yaw = np.arctan2(-My_comp, Mx_comp)

    # apply magnetic declination if given
    if declination_deg:
        yaw += np.deg2rad(declination_deg)

    # Normalize yaw to [-pi, pi)
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi

    # Compose rotation matrix R_body_to_world using 'xyz' intrinsic order: roll, pitch, yaw
    # Here we build R = R_z(yaw) @ R_y(pitch) @ R_x(roll) such that world = R @ body
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rotation matrices
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])

    # world = Rz * Ry * Rx * body
    R = Rz @ Ry @ Rx

    # Convert rotation matrix to quaternion [x, y, z, w] (w last)
    # Use robust algorithm:
    K = np.empty((4,4))
    K[0,0] = (R[0,0] - R[1,1] - R[2,2]) / 3.0
    K[0,1] = (R[1,0] + R[0,1]) / 3.0
    K[0,2] = (R[2,0] + R[0,2]) / 3.0
    K[0,3] = (R[1,2] - R[2,1]) / 3.0

    K[1,0] = (R[1,0] + R[0,1]) / 3.0
    K[1,1] = (R[1,1] - R[0,0] - R[2,2]) / 3.0
    K[1,2] = (R[2,1] + R[1,2]) / 3.0
    K[1,3] = (R[2,0] - R[0,2]) / 3.0

    K[2,0] = (R[2,0] + R[0,2]) / 3.0
    K[2,1] = (R[2,1] + R[1,2]) / 3.0
    K[2,2] = (R[2,2] - R[0,0] - R[1,1]) / 3.0
    K[2,3] = (R[0,1] - R[1,0]) / 3.0

    K[3,0] = (R[1,2] - R[2,1]) / 3.0
    K[3,1] = (R[2,0] - R[0,2]) / 3.0
    K[3,2] = (R[0,1] - R[1,0]) / 3.0
    K[3,3] = (R[0,0] + R[1,1] + R[2,2]) / 3.0

    # eigenvector of K corresponding to max eigenvalue is quaternion (x,y,z,w)
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    quat = np.array([q[0], q[1], q[2], q[3]])

    if degrees:
        return {
            'roll': np.degrees(roll),
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw),
            'R': R,
            'quat': quat
        }
    else:
        return {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'R': R,
            'quat': quat
        }
    
def acc_mag_to_rpy_ned(acc, mag, declination_deg=0.0):
    # Normalize
    Ax, Ay, Az = acc / np.linalg.norm(acc)
    Mx, My, Mz = mag / np.linalg.norm(mag)

    # Roll and pitch
    roll  = np.arctan2(Ay, Az)
    pitch = np.arctan2(-Ax, np.sqrt(Ay**2 + Az**2))

    # Tilt compensation
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Mx_comp = Mx * cp + Mz * sp
    My_comp = Mx * sr * sp + My * cr - Mz * sr * cp

    # Yaw (NED convention: clockwise positive)
    yaw = np.arctan2(My_comp, Mx_comp)
    yaw += np.deg2rad(declination_deg)
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi  # normalize to [-pi, pi)

    return roll, pitch, yaw

def acc_mag_to_R_body_to_ned(acc, mag, declination_deg=0.0):
    """
    Compute roll, pitch, yaw (NED) and rotation matrix R_body_to_NED from acc and mag.
    acc, mag : array-like shape (3,)
    declination_deg : magnetic declination to add to yaw (clockwise positive)
    Returns: dict with keys 'roll','pitch','yaw' (radians), 'R' (3x3), 'quat' ([x,y,z,w])
    """
    acc = np.asarray(acc, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if np.linalg.norm(acc) == 0 or np.linalg.norm(mag) == 0:
        raise ValueError("acc or mag has zero length")

    # Normalize sensors
    Ax, Ay, Az = acc / np.linalg.norm(acc)   # acc measured with +z down in NED
    Mx, My, Mz = mag / np.linalg.norm(mag)

    # Roll and pitch (NED)
    roll  = np.arctan2(Ay, Az)                        # +roll = right wing down
    pitch = np.arctan2(-Ax, np.sqrt(Ay*Ay + Az*Az))   # +pitch = nose up

    # Tilt-compensate magnetometer
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)

    Mx_comp = Mx * cp + Mz * sp
    My_comp = Mx * sr * sp + My * cr - Mz * sr * cp

    # Yaw (NED: clockwise positive, heading from North toward East)
    yaw = np.arctan2(My_comp, Mx_comp)
    yaw += np.deg2rad(declination_deg)
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi  # normalize to [-pi, pi)

    # Rotation matrices (about axes)
    Rx = np.array([[1, 0,    0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy := np.cos(yaw), - (sy := np.sin(yaw)), 0],
                   [sy, cy, 0],
                   [0, 0, 1]])

    # Compose: intrinsic roll -> pitch -> yaw  => R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    R = Rz @ Ry @ Rx

    # Convert R to quaternion [x,y,z,w] (w last) — robust method
    # Using standard conversion from rotation matrix
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        # find the largest diagonal element
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s

    quat = np.array([x, y, z, w])

    return {'roll': roll, 'pitch': pitch, 'yaw': yaw, 'R': R, 'quat': quat}
def kalman_filter_1d(z, Q=1e-5, R=0.01):
    n = len(z)
    x_hat = np.zeros(n)     
    P = np.zeros(n)         
    x_hat[0] = z[0]         
    P[0] = 1.0              

    for k in range(1, n):
        x_pred = x_hat[k-1]
        P_pred = P[k-1] + Q
        K = P_pred / (P_pred + R)
        x_hat[k] = x_pred + K * (z[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x_hat
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

def quat_inv_2(q):
    return [q[0],-q[1],-q[2],q[3]]
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
    """if mp.norm(qw)>10e-3:
        if qw<0:
            qw=-qw
            qx=-qx
            qy=-qy
            qz=-qz
        return np.array([qw,qx,qy,qz],dtype=mpf)"""
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

def intersection_line_from_planes(n1, d1, n2, d2):
    n1 = np.array(n1, dtype=float)
    n2 = np.array(n2, dtype=float)
    
    v = np.cross(n1, n2)
    
    if np.allclose(v, 0):
        raise ValueError("Planes are colinear.")
    
    A = np.array([n1, n2, v])
    b = -np.array([d1, d2, 0], dtype=float)
    
    P = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return P, v

a = np.random.random_sample(3)*np.pi
u = ExpRot(a)


q = ExpQua(a)


u = np.array([1,0,0])
v= np.array([0,1,1])

qu = ExpQua(u)
qv = ExpQua(v)
ru = ExpRot(u)
rv = ExpRot(v)