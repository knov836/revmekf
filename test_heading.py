import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt

from variants.integration_gyroaccmag import PredictFilter as IntegrationGyroAccMag
from variants.mekf_gyroaccmag import PredictFilter as MEKFGyroAccMag
from variants.reversible_gyroaccmag import PredictFilter as RevGyroAccMag

from variants.integration_gyroacc import PredictFilter as IntegrationGyroAcc
from variants.mekf_gyroacc import PredictFilter as MEKFGyroAcc
from variants.reversible_gyroacc import PredictFilter as RevGyroAcc

from variants.integration_odoaccpre import PredictFilter as IntegrationOdoAccPre
from variants.mekf_odoaccpre import PredictFilter as MEKFOdoAccPre
from variants.reversible_odoaccpre import PredictFilter as RevOdoAccPre

from mpmath import mp
from mpmath import mpf
import pandas as pd
import sympy as sp
from scipy.signal import savgol_filter
from matplotlib.markers import MarkerStyle
from pyproj import Proj

from pyproj import Transformer
from scipy.signal import butter, sosfiltfilt

from proj_func import correct_proj2
from function_quat import *
from signal_process import *
from kfilterdata2 import KFilterDataFile


mp.dps = 40
#mp.prec = 40

import sys,os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/examples')[0])


from scipy.spatial.transform import Rotation
from solver_kalman import SolverFilterPlan

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

g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)

data_file = 'selected_data_vehicle1.csv'

data=pd.read_csv(data_file)


#mmode = 'OdoAccPre'
mmode = 'GyroAccMag'


if mmode == 'GyroAcc':
    Integration = IntegrationGyroAcc
    MEKF = MEKFGyroAcc
    Rev = RevGyroAcc
if mmode == 'GyroAccMag':
    Integration = IntegrationGyroAccMag
    MEKF = MEKFGyroAccMag
    Rev = RevGyroAccMag
if mmode == 'OdoAccPre':
    Integration = IntegrationOdoAccPre
    MEKF = MEKFOdoAccPre
    Rev = RevOdoAccPre


n_start = 0
n_end=4000
n_end=n_start +10000
cols = np.array([0,1,2,3,10,11,12,19,20,21])
df = data.values[n_start:n_end,cols]

acc = np.copy(df[:,1:4])
mag = np.copy(df[:,7:10])

fs = 50
sos = butter(2, 24, fs=fs, output='sos')
#smoothed = sosfiltfilt(sos, newset.acc)
accs = np.copy(df[:,1:7])

df[:,4:7]=df[:,4:7]*np.pi/180

time= np.array(df[:,0],dtype=mpf)#/10**9
#time = time-2*time[0]+time[1]
df[:,0]*=10**9
#time = time-2*time[0]+time[1]#
#df[:,7:10] = c_mag

#normal = np.mean(df[:100,7:10],axis=0)



N = n_end-n_start
normals = np.zeros((N,3))
std_acc_zs = np.zeros(N)
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])



acc_smooth0 = savgol_filter(df[:,1], 500, 2)
acc_smooth1 = savgol_filter(df[:,2], 500, 2)
acc_smooth2 = savgol_filter(df[:,3], 500, 2)
acc_smooth = np.vstack((acc_smooth0,acc_smooth1,acc_smooth2)).T


acc_z = df[:,3]
s_acc_z = acc_z
s_acc_z = kalman_filter_1d(acc_z,10**(-2),0.1)
df[:,3] = s_acc_z


newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23,normals=normals)#,gravity=np.array([0,0,9.80665],dtype=mpf))#,normal=np.array([0.1101,1,0])) 
N=newset.size
#N=len(df)
nn = N-1
g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)
angle = int(N/2)

orient = newset.orient
pos_earth = newset.pos_earth

q0,q1,r0,r1 = 10**(-2), 10**(-10), 10**(0), 10**(-2)
normal = newset.normal


gravity = newset.gravity



proj_func = correct_proj2
proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)
Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)

newset.orient = Solv0.quaternion[:,:]  
nn=0

proj_utm = Proj(proj="utm", zone=31, ellps="WGS84")
gps = data.values[n_start:n_end,[-3,-2]]
R = 6371000
x, y = proj_utm(gps[:,1], gps[:,0])
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32722", always_xy=True)
x, y = transformer.transform(gps[:,1], gps[:,0])
coords = np.column_stack((x, y))-np.array([x[0],y[0]])



correction_applied = np.zeros(N)
angle_applied = np.zeros(N)
     
gravity = [0,0,np.mean(np.linalg.norm(acc_smooth[:150,:],axis=1))]

q2 = np.zeros((N,3))
per = 500
speed = np.diff(coords[::per,:].astype(float),axis=0)
theta = np.arctan2(speed[:,1],speed[:,0])
theta = np.where(theta  < 0, theta + 2 * np.pi, theta)

for i in range(N):
    q2[i,:] = np.array(quat_rot([0,1,0,0],(newset.neworient[i,:])))[1:4]
    
    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q2)
ax.set_title('q2')

theta_q2 = np.arctan2(q2[::per,0],q2[::per,1])
theta_q2 = np.where(theta_q2 < 0, theta_q2 + 2 * np.pi, theta_q2 )

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(theta)
ax.plot(theta_q2)
ax.set_title("compare theta q2")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.mag)


from scipy.spatial.transform import Rotation as R

# Example: array of N quaternions (each in [x, y, z, w] format)
quats = np.array([
    [0, 0, 0.7071, 0.7071],
    [0.5, 0.5, 0.5, 0.5],
    [0, 0, 0, 1],
])



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(theta)
ax.set_title("GPS heading")



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(newset.mag[:,1].astype(float),newset.mag[:,0].astype(float)))
ax.set_title("Mag heading")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(q2[:,1],q2[:,0]))
ax.set_title("q2 heading")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.linalg.norm(np.cross(newset.acc[:100,:].astype(float),newset.mag[:100,:].astype(float)),axis=1))

a = newset.acc[0,:].astype(float)
a=a/np.linalg.norm(a)
m = newset.mag[0,:].astype(float)

adm = np.cross(a,m)
nm = np.cross(adm,a)
print(nm,m)


res = acc_mag_to_R_body_to_ned(a, m, declination_deg=0.0)
qq2 = res['R']@ np.array([1,0,0])
print(np.arctan2(qq2[1],qq2[0]))
print(acc_mag_to_rpy_ned(a,m))

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.neworient)
ax.set_title('orient')