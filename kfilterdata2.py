
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numdifftools as nd 
import pdb

from mpmath import mp
from mpmath import mpf
from function_quat import *
import pandas as pd
import sys,os



grav=1
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

class KFilterDataFile:
    def __init__(self, data,mode='GyroAccMag',g_bias = np.array([0,0,0],dtype=mpf),base_width=mpf(1.0),surf=np.array([-1,1,0],dtype=mpf),normal=np.array([]),gravity=np.array([]),normals=np.array([]),start=np.array([])):
        gyro = np.array(data[:,4:7],dtype=mpf)#/180*np.pi
        acc = np.array(data[:,1:4],dtype=mpf)
        mag = np.array(data[:,7:10],dtype=mpf)
        time = np.array(data[:,0],dtype=mpf)/10**9
        self.base_width=base_width
        self.mode = mode
        self.normal = normal
        dtime = np.diff(time)
        #print(dtime)
        
        ind = np.where(1/dtime<1000)[0]
        #print(data[])
        
        self.freq = np.mean(1/dtime[ind])
        self.size=len(time)
        self.c_size = 10
        self.DT = mpf(1)/mpf(self.freq)
        self.grav = np.mean([mp.norm(a) for a in acc])
        
        
        
        self.time = time
        #self.gyro = gyro
        self.pos_earth = np.zeros((self.size,3))
        self.speed_earth = np.zeros((self.size,3))
        self.acc_earth = np.zeros((self.size,3))
        self.mag_earth = np.zeros((self.size,3))
        self.a_noise = 0
        self.m_noise = 0
        self.acc = acc
        self.mag = mag
        self.g_bias = np.mean(gyro[:self.c_size,:],axis=0)
        self.gyro = np.copy(gyro-self.g_bias)
        self.c_mag = np.copy(self.cmag(normal=self.normal))
        calib = self.new_orient_calib()
        if(len(start) ==0):
            quat_calib = np.mean(calib[:self.c_size,:],axis=0)
        else:
            quat_calib = start
        
        self.quat_calib = quat_calib/np.linalg.norm(quat_calib)
        
        
        perceived_gravity = -np.mean(self.acc[:self.c_size,:],axis=0)
        self.pg_std = np.std(self.acc[:100,:].astype(float),axis=0)
        n_perceived_gravity = perceived_gravity/np.linalg.norm(perceived_gravity)
        qq = quat_ntom(n_perceived_gravity, np.array([0,0,1],dtype=mpf))
        if len(gravity) == 0:
            self.gravity = np.array(quat_rot([0,*perceived_gravity],qq))[1:4]#+0.01#-np.array([0,0,np.linalg.norm(self.pg_std)],dtype=mpf)/10
        else:
            self.gravity=gravity
        
        
        if len(self.normal)==0:
            self.normal = np.array(quat_rot([0,0,0,1],(self.quat_calib)))[1:4]
            normal = self.normal
            
        m_mag = np.mean(mag[:self.c_size,:],axis=0)
        
        self.d_mag = np.array(quat_rot([0,*m_mag],self.quat_calib))[1:4]
        self.d_mag = self.d_mag/np.linalg.norm(self.d_mag)
        oo = skewSymmetric(self.normal)@self.d_mag
        plane_mag = oo/np.linalg.norm(oo)
        self.plane_mag = plane_mag
        
        new_normal = skewSymmetric(self.d_mag)@plane_mag
        self.new_normal = new_normal
        east = np.array([0,1,0],dtype=mpf)
        orth = skewSymmetric(self.normal)@east
        orth=orth/np.linalg.norm(orth)
        
        target_vector = -skewSymmetric(new_normal)@orth
        target_vector = target_vector/np.linalg.norm(target_vector)
        
        qq0 = quat_ntom(self.d_mag, target_vector)
        beta_rot_mag = log_q(qq0)
        self.beta_mag = np.linalg.norm(beta_rot_mag)*np.sign(np.dot(new_normal,beta_rot_mag))
        
        assert(np.linalg.norm(beta_rot_mag- self.beta_mag*new_normal)<10**(-10))
        
        r_plane_mag = east
        self.orth = orth
        self.target_vector = target_vector
        self.r_plane_mag = r_plane_mag
        l_rot_mag = log_q(quat_ntom(target_vector, east))
        self.alpha_mag = np.linalg.norm(l_rot_mag)*np.sign(np.dot(orth,l_rot_mag))
        assert(np.linalg.norm(l_rot_mag - self.alpha_mag*orth)<10**(-10))
        
        self.mag = np.copy(self.omag(self.gravity))
        self.neworient = self.new_orient()
            
        self.rotsurf = quat_ntom(np.array([0,0,1]),normal)
        self.surf = np.array(np.concatenate(([0,],normal.astype(mpf),np.zeros(6))),dtype=mpf)
        
        
        
        self.orient = np.copy(self.neworient)
        self.orient[0,:] = self.quat_calib
        
        
        self.mag0 = np.array(quat_rot([0,*np.mean(self.mag[:10,:].astype(float),axis=0)], self.quat_calib))[1:4].astype(float)
        
        self.mag0 = np.array(quat_rot([0,*self.mag0],ExpQua(np.array([0,0,-np.arctan2(self.mag0[1],self.mag0[0])]))))[1:4]
        #pdb.set_trace()
        self.mag0 = self.mag0/np.linalg.norm(self.mag0)
        #self.mag0 = np.array(quat_rot([0,*np.mean(self.mag[:300,:].astype(float),axis=0)], quat_inv(self.quat_calib)))[1:4]
        
        #self.mag0 = np.array([1,0,0],dtype=mpf)
        
    def cmag(self,normal=None):
        cmag= np.zeros((self.c_size,3),dtype=mpf)
        for i in range(self.c_size):
            a=self.acc[i,:]
            a=a/np.linalg.norm(a)
            m=self.mag[i,:]
            if mp.norm(m)!=0:
                m=m/mp.norm(m)
            cmag[i,:] = m
            continue
            if mp.norm(m)!=0:
                m=m/mp.norm(m)
                adm = skewSymmetric(a)@m
                adm = adm/mp.norm(adm)
                new_m = skewSymmetric(adm)@a
                cmag[i,:] = new_m
            #print(m,new_m)
        
        return cmag
    def omag(self,gravity):
        omag= np.zeros((self.size,3),dtype=mpf)
        for i in range(self.size):
            
            m=self.mag[i,:]
            if mp.norm(m)!=0:
                m=m/mp.norm(m)
            omag[i,:] = m
            continue
            if mp.norm(m)!=0:
                a=self.acc[i,:]
                a = a/np.linalg.norm(a)
                """a[2] = a[2] /np.linalg.norm(gravity)
                alpha = np.sqrt(np.max(np.abs(1-a[2]**2),0))
                a[0:2] = a[0:2]/np.linalg.norm(a[0:2])*alpha"""
                
                m=m/mp.norm(m)
                adm = skewSymmetric(a)@m
                adm = adm/mp.norm(adm)
                new_new_m= skewSymmetric(adm)@a
                
                omag[i,:] = new_new_m
        
        return omag

    
    def calc_speed_earth(self,pos_earth):
        speed_ref = np.vstack([np.zeros(3,dtype=mpf),np.diff(pos_earth,axis=0)])/self.DT
        return speed_ref
    def calc_acc_earth(self,speed_earth):
        acc_earth = np.vstack([np.zeros(3,dtype=mpf),np.diff(speed_earth,axis=0)])/self.DT+np.array(self.gravity)

        acc= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            acc[i,2]=1

        acc[:acc_earth.shape[0]] = acc_earth
        return acc
    
    def calc_mag_earth(self):
        mag= np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            mag[i,1]=1
        return mag
    
    def new_orient_calib(self):
        new_orient = np.zeros((self.c_size,4),dtype=mpf)
        for i in range(self.c_size):
            a = self.acc[i,:]
            a = a/mp.norm(a)
            a0 = np.array([0,0,1],dtype=mpf)
            #a = -a0
            m = self.c_mag[i,:]
            m = m/mp.norm(m)
            m0 = np.array([0,1,0],dtype=mpf)
            
            adm = skewSymmetric(a)@m
            adm = adm/mp.norm(adm)
            new_m = skewSymmetric(adm)@a
            #new_m = m
            M = np.array([-adm,new_m,a]).T
            M = np.array([new_m,adm,a]).T
            #print("new_m",new_m,m)
            """M = np.array([new_m,adm,a]).T
            res = acc_mag_to_R_body_to_ned(a, m, declination_deg=0.0)
            M = res['R']
            res = acc_mag_to_rotation(a, m, declination_deg=0.0)
            M = res['R']"""
            #print(res['yaw'])
            
            new_orient[i,:] = normalize(quat_inv(RotToQuat(M)))
            #new_orient[i,:] = normalize((RotToQuat(M)))
        return new_orient
    def new_orient(self):
        new_orient = np.zeros((self.size,4),dtype=mpf)
        for i in range(self.size):
            a = self.acc[i,:]
            a = a/mp.norm(a)
            a0 = np.array([0,0,1],dtype=mpf)
            #a = -a0
            m = self.mag[i,:]
            m = m/mp.norm(m)
            m0 = np.array([0,1,0],dtype=mpf)
            adm = skewSymmetric(a)@m
            adm = adm/mp.norm(adm)
            new_m = skewSymmetric(adm)@a
            
            M = np.array([-adm,new_m,a]).T
            #pdb.set_trace()
            M = np.array([new_m,adm,a]).T
            """M = np.array([new_m,adm,a]).T
            res = acc_mag_to_R_body_to_ned(a, m, declination_deg=0.0)
            M = res['R']
            res = acc_mag_to_rotation(a, m, declination_deg=0.0)
            M = res['R']"""
            #print("here",M@[0,0,1],a)
            """print("here a",[1,0,0]@(M.T),m)
            print("here b",[0,1,0]@(M.T),m)"""
            
            new_orient[i,:] = normalize(quat_inv(RotToQuat(M)))
            #print("toto",np.array(quat_rot([0,*(m.astype(float))], (new_orient[i,:])))[1:4])
            #print(normalize(quat_inv(RotToQuat(M))))
        return new_orient
    
    def mems_ref(self,tab):
        res = np.zeros((len(self.orient),3),dtype=mpf)
        for i in range(len(self.orient)):
            
            dd_quat = np.array([0,tab[i,0],tab[i,1],tab[i,2]],dtype=mpf)
            conjbbi = np.array([-self.orient[i,0],self.orient[i,1],self.orient[i,2],self.orient[i,3]],dtype=mpf)
            quat = quat_rot(dd_quat,conjbbi)
            res[i,:] = quat[1:4]
        return res
    
N=100
angle = int(N/2)



data_file = 'test1.csv'
data_file = 'calibration_data_250523.csv'
data_file = 'imu_data_sbg.csv'
mag_file = 'mag_data_sbg.csv'
#N = 10000
data=pd.read_csv(data_file)
mag=pd.read_csv(mag_file)
n_start = 0
n_end=1200
df = data.values[n_start:n_end,:]
c_mag = mag.values[n_start:n_end,:]

newset = KFilterDataFile(df[1:,:],surf=np.array([0,0,1])) 

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.pos_earth)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient-newset.neworient)
ax.set_title("diff acc and computed orient")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc[:,:])
ax.set_title("acc")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.mag[:,:])
ax.set_title("mag")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(c_mag[:,:])
ax.set_title("c_mag")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient[:,0],newset.orient[:,3],"x")
ax.plot(newset.neworient[:,0],newset.neworient[:,3],"x")
ax.set_title("compare orient")

R = QuatToRot(newset.neworient[10,:])
