import numpy as np
from scipy import *
from scipy.spatial.transform import Rotation as R
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

from scipy.io import loadmat



mp.dps = 40
#mp.prec = 40

import sys,os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/examples')[0])


from scipy.spatial.transform import Rotation
from solver_kalman import SolverFilterPlan

acc_columns_x = [
    'x',
    #'acc_x_above_suspension',
    #'acc_x_below_suspension'
]
acc_columns_y = [
    'y',
    #'acc_y_above_suspension',
    #'acc_y_below_suspension'
]
acc_columns_z = [
    'z',
    #'acc_z_above_suspension',
    #'acc_z_below_suspension'
]
gyro_columns_x = [
    'x'#,
    #'gyro_x_above_suspension'#,'gyro_x_below_suspension'
]
gyro_columns_y = [
    'y'#,
    #'gyro_y_above_suspension'#,'gyro_y_below_suspension'
]
gyro_columns_z = [
    'z'#,
    #'gyro_z_above_suspension'#,'gyro_z_below_suspension'
]
mag_columns_x = [
    'x',
    #'mag_x_above_suspension'#,'mag_x_below_suspension'
]
mag_columns_y = [
    'y',
    #'mag_y_above_suspension'#,'mag_y_below_suspension'
]
mag_columns_z = [
    'z',
    #'mag_z_above_suspension'#,'mag_z_below_suspension'
]
acc_x = len(acc_columns_x)
acc_y = len(acc_columns_y)
acc_z = len(acc_columns_z)
gyro_x = len(gyro_columns_x)
gyro_y = len(gyro_columns_y)
gyro_z = len(gyro_columns_z)
mag_x = len(mag_columns_x)
mag_y = len(mag_columns_y)
mag_z = len(mag_columns_z)


sentral = loadmat('V1000/Sequencia1/SENTRAL.mat')
ori_abb = sentral['ABBquaternion'].T
ori = sentral['SENSquat'].T
ori_z = np.copy(ori[:,1])
ori[:,1] = np.copy(ori[:,2])
ori[:,2] = ori_z

gps_v = sentral['ABBposition'][:2,:].T
sentral_time = sentral['MATLABtime'].flatten()
MPU9150 = loadmat('V1000/Sequencia1/MPU9150.mat')
MPU6050RM3100 = loadmat('V1000/Sequencia1/MPU6050RM3100.mat')


grav = 9.80
acc_v = MPU9150['SENSacc'].T*grav
gyro_v = MPU9150['SENSgyro'].T*np.pi/180
mag_v = MPU9150['SENSmag'].T
mag_v0 = np.copy(mag_v)
mag_v[:,0] = mag_v0[:,1]
mag_v[:,1] = mag_v0[:,0]
mag_v[:,2] = -mag_v0[:,2]
time = MPU9150['MATLABtime'].flatten()

acc_df = pd.DataFrame({
    'timestamp': time,
    'acc_x': acc_v[:,0],
    'acc_y': acc_v[:,1],
    'acc_z': acc_v[:,2]
}).sort_values('timestamp')

acc_df = acc_df.dropna()
acc_df = acc_df.drop(acc_df.index[0])

gyro_df = pd.DataFrame({
    'timestamp': time,
    'gyro_x': gyro_v[:,0],
    'gyro_y': gyro_v[:,1],
    'gyro_z': gyro_v[:,2]
}).sort_values('timestamp')
gyro_df = gyro_df.dropna()
gyro_df = gyro_df.drop(gyro_df.index[0])

mag_df = pd.DataFrame({
    'timestamp': time,
    'mag_x': mag_v[:,0],
    'mag_y': mag_v[:,1],
    'mag_z': mag_v[:,2]
}).sort_values('timestamp')
mag_df = mag_df.dropna()
"""'ori_w': ori[:,0],
'ori_x': ori[:,1],
'ori_y': ori[:,2],
'ori_z': ori[:,3],"""
gps_df = pd.DataFrame({
    'timestamp': sentral_time,#*10**9,  # ou ton propre vecteur GPS timestamps
    'gps_x': gps_v[:,0],
    'gps_y': gps_v[:,1],
    'ori_x': ori[:,0],
    'ori_y': ori[:,1],
    'ori_z': ori[:,2],
    'ori_w': ori[:,3],
    #'speed': gps[:,2] if gps.shape[1] > 2 else np.nan
}).sort_values('timestamp')
gps_df = gps_df.dropna()
mpu = pd.merge_asof(acc_df, gyro_df, on='timestamp', direction='nearest')
mpu = pd.merge_asof(mpu, mag_df, on='timestamp', direction='nearest')
mpu = pd.merge_asof(mpu, gps_df, on='timestamp', direction='nearest')

mpu = mpu.dropna(how="all")
print(mpu.head())

g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)

#data_file = 'selected_data_vehicle1.csv'

data=mpu


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
n_end=n_start +1500
cols = np.array([0,1,2,3,10,11,12,19,20,21])
cols = np.array([0,7,8,9,16,17,18,25,26,27])
cols = np.array(range(10))
df = data.values[n_start:n_end,cols]

acc = np.copy(df[:,1:4])
mag = np.copy(df[:,7:10])

fs = 50
sos = butter(2, 24, fs=fs, output='sos')
#smoothed = sosfiltfilt(sos, newset.acc)
accs = np.copy(df[:,1:7])

#df[:,4:7]=df[:,4:7]*np.pi/180
gyro = np.copy(df[:,4:7])
time= np.array(df[:,0],dtype=mpf)#/10**9
#time = time-2*time[0]+time[1]
df[:,0]*=10**9
#time = time-2*time[0]+time[1]#
#df[:,7:10] = c_mag

"""df[:,1] = accs[:,1]
df[:,2] = accs[:,0]
df[:,3] = -accs[:,2]
df[:,4] = gyro[:,1]
df[:,5] = gyro[:,0]
df[:,6] = -gyro[:,2]"""

"""df[:,1] = accs[:,2]
df[:,2] = -accs[:,1]
df[:,3] = accs[:,0]
df[:,4] = gyro[:,2]
df[:,5] = -gyro[:,1]
df[:,6] = gyro[:,0]
"""
#df[:,2] = -accs[:,1]
#df[:,3] = -accs[:,2]
"""df[:,4] = -gyro[:,2]
df[:,6] = -gyro[:,0]"""
#normal = np.mean(df[:100,7:10],axis=0)
def Calibrate_Mag(magX, magY, magZ):
    x2 = (magX ** 2)
    y2 = (magY ** 2)
    z2 = (magZ ** 2)
    yz = 2*np.multiply(magY, magZ)
    xz = 2*np.multiply(magX, magZ)
    xy = 2*np.multiply(magX, magY)
    x = 2*magX
    y = 2*magY
    z = 2*magZ
    d_tmp = np.ones(len(magX))
    d = np.expand_dims(d_tmp, axis=1)
    print(len(d),x2.shape,y2.shape,yz.shape,x.shape,d.shape)
    D = np.array([x2, y2, z2, yz, xz, xy, x, y, z, d])
    D = D[:,:, 0]
    C1 = np.array([[-1, 1, 1, 0, 0, 0],
                   [1, -1, 1, 0, 0, 0],
                   [1, 1, -1, 0, 0, 0],
                   [0, 0, 0, -4, 0, 0],
                   [0, 0, 0, 0, -4, 0],
                   [0, 0, 0, 0, 0, -4]])
    
    # Equation 11 --- S = D(D.T)
    #D_T = np.transpose(D, (1, 0, 2))
    S = np.matmul(D, D.T)
    print("S Shape: ", S.shape)
    S11 = S[:6, :6]
    S12 = S[:6, 6:]
    S21 = S[6:, :6]
    S22 = S[6:, 6:]
    S22 = S22.astype(float)
    print(S22.shape)
    # Equation 15
    print(np.linalg.inv(S22.astype(float)))
    tmp1 = np.matmul(S12, np.matmul(np.linalg.inv(S22), S12.T))
    tmp = np.matmul(np.linalg.inv(C1), S11 - tmp1)
    eigenValue, eigenVector = np.linalg.eig(tmp.astype(float))
    v1 = eigenVector[:, np.argmax(eigenValue)]
    if v1[0] < 0: v1 = -v1
    
    # Equation 13
    v2 = np.matmul(-np.matmul(np.linalg.inv(S22), S12.T), v1)
    
    # Equation 11 (part 2)
    v = np.concatenate([v1, v2]).T
    
    M = np.array([[v[0], v[5], v[4]],
                  [v[5], v[1], v[3]],
                  [v[4], v[3], v[2]]])
    
    n = np.array([[v[6]],
                  [v[7]],
                  [v[8]]])
    d = v[9]
    
    Minv = np.linalg.inv(M)
    b = -np.dot(Minv, n)
    Ainv = np.real(1 / np.sqrt(np.dot(n.T, np.dot(Minv, n)) - d) * linalg.sqrtm(M))

    return Minv, b, Ainv


def Calibrate_Mag_improved(magX, magY, magZ, reg=1e-8, refine=False, refine_maxiter=200):
    # ensure 1D arrays and double precision
    magX = np.asarray(magX, dtype=np.float64).reshape(-1)
    magY = np.asarray(magY, dtype=np.float64).reshape(-1)
    magZ = np.asarray(magZ, dtype=np.float64).reshape(-1)
    N = magX.size
    if N < 10:
        raise ValueError("Not enough samples for stable calibration. Need >= 10 samples.")

    # build D rows (10 x N)
    x2 = magX**2
    y2 = magY**2
    z2 = magZ**2
    yz = 2 * magY * magZ
    xz = 2 * magX * magZ
    xy = 2 * magX * magY
    x = 2 * magX
    y = 2 * magY
    z = 2 * magZ
    ones = np.ones(N, dtype=np.float64)

    D = np.vstack([x2, y2, z2, yz, xz, xy, x, y, z, ones])  # shape (10, N)

    # S = D * D^T (10 x 10)
    S = D @ D.T
    # partition
    S11 = S[:6, :6]
    S12 = S[:6, 6:]
    S21 = S[6:, :6]
    S22 = S[6:, 6:]

    # regularize S22 for stability
    S22 = S22.astype(np.float64) + reg * np.eye(S22.shape[0], dtype=np.float64)

    # constraint matrix C1 (6x6)
    C1 = np.array([[-1, 1, 1, 0, 0, 0],
                   [1, -1, 1, 0, 0, 0],
                   [1, 1, -1, 0, 0, 0],
                   [0, 0, 0, -4, 0, 0],
                   [0, 0, 0, 0, -4, 0],
                   [0, 0, 0, 0, 0, -4]], dtype=np.float64)

    # Equation 15 derivation: tmp = inv(C1) * (S11 - S12 * inv(S22) * S12^T)
    # use solve instead of explicit inverse
    tmp1 = S12 @ np.linalg.solve(S22, S12.T)
    mat_for_eig = S11 - tmp1

    # Solve C1 * X = mat_for_eig  => X = inv(C1) * mat_for_eig
    X = np.linalg.solve(C1, mat_for_eig)

    # symmetric eigensolve (we expect symmetric)
    eigvals, eigvecs = np.linalg.eig(X)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # choose eigenvector corresponding to largest eigenvalue
    idx = np.argmax(eigvals)
    v1 = eigvecs[:, idx]
    if v1[0] < 0:
        v1 = -v1

    # v2 from Eq 13: v2 = -inv(S22) * S12^T * v1
    v2 = -np.linalg.solve(S22, S12.T @ v1)

    # combined parameter vector v (10,)
    v = np.hstack([v1, v2])

    # build M (3x3), n (3x1), d (scalar)
    M = np.array([[v[0], v[5], v[4]],
                  [v[5], v[1], v[3]],
                  [v[4], v[3], v[2]]], dtype=np.float64)
    n = np.array([v[6], v[7], v[8]], dtype=np.float64).reshape(3, 1)
    d = float(v[9])

    # invert M
    Minv = np.linalg.inv(M)

    # bias b
    b = -Minv @ n
    b = b.reshape(3)  # 1D

    # compute Ainv to normalize to unit sphere:
    # Ainv = (1/sqrt(n^T Minv n - d)) * sqrtm(M)
    # use scipy.linalg.sqrtm if available, else eigen-decompose M
    try:
        from scipy import linalg as sp_linalg
        sqrtM = sp_linalg.sqrtm(M)
    except Exception:
        w, U = np.linalg.eigh(M)
        w = np.maximum(w, 0)
        sqrtM = (U * np.sqrt(w)) @ U.T

    scalar = 1.0 / np.sqrt(float(n.T @ Minv @ n) - d)
    Ainv = scalar * np.real(sqrtM)

    if refine:
        # refine by optimizing T (3x3) and b (3,) so that ||T @ (m - b)|| == 1
        try:
            from scipy.optimize import least_squares
        except Exception as e:
            raise RuntimeError("scipy is required for refinement: pip install scipy") from e

        # parameterize T as 9 params and b as 3 params -> 12 params
        T0 = Ainv  # good initial guess
        p0 = np.hstack([T0.ravel(), b.ravel()])

        Mdata = np.vstack([magX, magY, magZ])  # 3 x N

        def residuals(p):
            T = p[:9].reshape(3, 3)
            bb = p[9:].reshape(3, 1)
            cal = T @ (Mdata - bb)  # 3 x N
            norms = np.linalg.norm(cal, axis=0)
            return (norms - 1.0)

        res = least_squares(residuals, p0, max_nfev=refine_maxiter, xtol=1e-10, ftol=1e-10)
        p_opt = res.x
        T_opt = p_opt[:9].reshape(3, 3)
        b_opt = p_opt[9:].reshape(3)
        Ainv = T_opt
        b = b_opt

    return Minv, b, Ainv

mags = np.array([mpu['mag_x'].values,mpu['mag_y'].values,mpu['mag_z'].values]).T
N = len(mags[:,0])
mX,mY,mZ = mags[:,0].reshape((N,1)),mags[:,1].reshape((N,1)),mags[:,2].reshape((N,1))
Minv, b, Ainv = Calibrate_Mag_improved(mX, mY, mZ)
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mX, mY, mZ, s=5, color='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)
ax.plot_surface(x, y, z, alpha=0.3, color='b')




calibratedX = np.zeros(mags[:,0].shape)
calibratedY = np.zeros(mags[:,1].shape)
calibratedZ = np.zeros(mags[:,2].shape)


c_mag = np.zeros(mags[:,:].shape)
totalError = 0
for i in range(len(c_mag)):
    h = np.array([np.array([mags[:,0][i], mags[:,1][i], mags[:,2][i]]).flatten().reshape((3,1))])
    hHat = np.matmul(Ainv, h-b)
    hHat = hHat[:, :, 0]
    calibratedX[i] = hHat[0][0]
    calibratedY[i] = hHat[0][1]
    calibratedZ[i] = hHat[0][2]
    mag0 = np.dot(hHat.T, hHat)
    err = (mag0[0][0] - 1)**2
    totalError += err
print("Total Error: %f" % totalError)
"""mag_unit = mags / np.linalg.norm(mags, axis=1, keepdims=True)
calibratedX,calibratedY,calibratedZ = mag_unit.T"""
c_mag[:,0] = calibratedX.flatten()
c_mag[:,1] = calibratedY.flatten()
c_mag[:,2] = calibratedZ.flatten()



fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(calibratedX , calibratedY , calibratedZ , s=5, color='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)
ax.plot_surface(x, y, z, alpha=0.3, color='b')


df[:,7:10] = c_mag[n_start:n_end,:]
mag = np.copy(df[:,7:10])
"""df[:,7] = mag[:,1]
df[:,8] = mag[:,0]
df[:,9] = -mag[:,2]
"""

N = n_end-n_start
normals = np.zeros((N,3))
std_acc_zs = np.zeros(N)
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])



acc_smooth0 = savgol_filter(df[:,1], 20, 2)
acc_smooth1 = savgol_filter(df[:,2], 20, 2)
acc_smooth2 = savgol_filter(df[:,3], 20, 2)
acc_smooth = np.vstack((acc_smooth0,acc_smooth1,acc_smooth2)).T


acc_z = df[:,3]
s_acc_z = acc_z
#s_acc_z = kalman_filter_1d(acc_z,10**(-2),0.1)
df[:,3] = s_acc_z

"""df[:,1:4] = np.copy(acc_smooth)
df[:,4:7] = df[:,4:7]-np.mean(gyro_v[:2000,:],axis=0)
"""
"""gyro_smooth0 = savgol_filter(df[:,4], 20, 2)
gyro_smooth1 = savgol_filter(df[:,5], 20, 2)
gyro_smooth2 = savgol_filter(df[:,6], 20, 2)
gyro_smooth = np.vstack((gyro_smooth0,gyro_smooth1,gyro_smooth2)).T
df[:,4:7] = np.copy(gyro_smooth)"""

#g_bias = np.mean(gyro_v[:2000,:],axis=0)
newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23,normals=normals,gravity=np.array([0,0,grav],dtype=mpf))#,start=np.array(q_ori[0,:],dtype=mpf))#,gravity=np.array([0,0,9.76],dtype=mpf))#,normal=np.array([0.1101,1,0])) 
N=newset.size
#N=len(df)
nn = N-1
g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)
angle = int(N/2)

orient = newset.orient
pos_earth = newset.pos_earth

q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(0), 10**(0)
normal = newset.normal


gravity = newset.gravity





proj_func = correct_proj2
proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)
Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)

newset.orient = Solv0.quaternion[:,:]  
nn=0

#proj_utm = Proj(proj="utm", zone=31, ellps="WGS84")
x,y= mpu[['gps_x','gps_y']].values[n_start:n_end,[0,1]].T
coords = np.column_stack((x, y))-np.array([x[0],y[0]])



correction_applied = np.zeros(N)
angle_applied = np.zeros(N)
     
gravity = [0,0,np.mean(np.linalg.norm(acc_smooth[:150,:],axis=1))]

q_ori = mpu[['ori_x','ori_y','ori_z','ori_w']].values[n_start:n_end,:]

quats_ori = np.array([log_q(np.array(quat_mult(RotToQuat(R.from_quat(q_ori[i,:]).as_matrix()),quat_inv(RotToQuat(R.from_quat(q_ori[0,:]).as_matrix()))))) for i in range(len(q_ori))])
xs = np.copy(quats_ori)
"""quats_ori[:,1]=-np.copy(quats_ori[:,1])
#quats_ori[:,0]=xs
quats_ori[:,0]=-np.copy(quats_ori[:,0])"""
quats_ori[:,2] = -xs[:,2]
quats_ori[:,1] = xs[:,1]
quats_ori[:,0] = -xs[:,0]
quats_ori[:,0] = -xs[:,1]
quats_ori[:,1] = -xs[:,2]
quats_ori[:,2] = xs[:,0]

Qt = np.mean([quat_mult(ori_abb[i,:],quat_inv(q_ori[i,:])) for i in range(10)],axis=0)
Qt = Qt/np.linalg.norm(Qt)

ref0 = np.array([quat_mult(q_ori[i,:],Qt) for i in range(len(q_ori))])
ref = np.array([quat_mult(ExpQua(quats_ori[i,:]),newset.quat_calib) for i in range(len(quats_ori))])
normals = np.zeros((N,3))
for i in range(0,N,1):
    normals[i] = np.array(quat_rot([0,0,1,0],ref[i,:]))[1:4]

    
ref_mag = np.zeros((N,3))

for i in range(0,N,1):
    ref_mag[i] = np.array(quat_rot([0,*newset.mag0],quat_inv(ref[i,:])))[1:4]

ref_acc = np.zeros((N,3))

for i in range(0,N,1):
    ref_acc[i] = np.array(quat_rot([0,0,0,1],quat_inv(ref[i,:])))[1:4]


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot([np.dot(newset.acc[i,:],newset.mag[i,:]) for i in range(N)])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot([np.dot(ref_acc[i,:],newset.mag[i,:]) for i in range(N)])


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.mag)
ax.plot(ref_mag)
ax.set_title('diff mag')
    

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)
ax.plot(ref_acc*np.linalg.norm(newset.gravity))
ax.set_title('diff acc')

for i in range(0,N-1,1):
    
    nn+=1
    if i < 850:
        normal = np.array(quat_rot([0,0,1,0],ref[i+1,:]))[1:4]
    elif i<1200:
        normal = np.array(quat_rot([0,1,0,0],ref[i+1,:]))[1:4]
    else:
        normal = np.array(quat_rot([0,0,0,1],ref[i+1,:]))[1:4]
    std_acc_z =0
    if i<N-1-40 and i> 40:
        std_acc_z = np.std(newset.acc[i-20:i+20,2])
    std_acc_zs[i+1] = std_acc_z 
    #print(std_acc_z)
    print("iteration",i)
    acci = np.array(quat_rot([0,0,0,1],quat_inv(ref[i+1,:])))[1:4]
    magi = np.array(quat_rot([0,*newset.mag0],quat_inv(ref[i+1,:])))[1:4]
    ref_mag[i+1,:] = magi
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    #Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal,std_acc_z=std_acc_z)
    correction_applied[i] = Solv2.KFilter.corrected
    angle_applied[i+1] =angle_applied[i]+Solv2.KFilter.angle

    if i%10 ==0 and i>0:
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot([np.linalg.norm(Solv0.position[j,:]) for j in range(i+1)])
        ax.plot([np.linalg.norm(Solv1.position[j,:]) for j in range(i+1)])
        ax.plot([np.linalg.norm(Solv2.position[j,:]) for j in range(i+1)])        
        ax.plot([np.linalg.norm(coords[j,:]) for j in range(i+1)])
        ax.plot(np.argwhere(correction_applied).flatten(), [np.linalg.norm(Solv2.position[j,:])  for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))
        plt.show()
        
compare = np.zeros((N,4),dtype=float)
compare2 = np.zeros((N,4),dtype=float)
quaternion0 = Solv0.quaternion[:N,:]    
quaternion1 = Solv1.quaternion[:N,:]
quaternion2 = Solv2.quaternion[:N,:]
position0 = Solv0.position[:N,:]
position1 = Solv1.position[:N,:]
position2 = Solv2.position[:N,:]
gravity_r = Solv2.gravity_r[:N,:]
time0=time-time[0]
size  =n_end-n_start


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(time0[:size-1],newset.acc[1:size,2])
ax.plot(time0[:size-1],gravity_r[1:size,2]*Solv2.KFilter.gravity[2])
ax.set_xlabel('Seconds')
ax.set_ylabel('m.s^(-2)')
ax.legend(['Z coordinate of Accelerometer','Z coordinate of gravity vector returned by Rev-MEKF'])
ax.set_title('Comparison of z coordinates for different acceleration computed')

mpu_abb = MPU9150['ABBquaternion'].T
Qt = np.mean([quat_mult(mpu_abb[i,:],quat_inv(quaternion1[i,:])) for i in range(5)],axis=0)
Qt = Qt/np.linalg.norm(Qt)

quaternion1_0 = np.array([quat_mult(quaternion1[i,:],Qt) for i in range(len(quaternion1))])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion0)
ax.set_title('quaternion0')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion1)
ax.set_title('quaternion1')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion2)
ax.set_title('quaternion2')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(ref)
ax.set_title('ref')





quat2 = np.array([log_q(np.array(quat_mult(quaternion2[i,:],quat_inv(quaternion2[0,:])))) for i in range(len(quaternion1))])
quat1 = np.array([log_q(np.array(quat_mult(quaternion1[i,:],quat_inv(quaternion1[0,:])))) for i in range(len(quaternion1))])
quat0 = np.array([log_q(np.array(quat_mult(quaternion0[i,:],quat_inv(quaternion0[0,:])))) for i in range(len(quaternion0))])

quat_accmag = np.array([log_q(np.array(quat_mult(newset.neworient[i,:],quat_inv(newset.neworient[0,:])))) for i in range(len(newset.neworient))])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quats_ori)
ax.set_title("quats_ori")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat2)
ax.set_title("quat2")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat1)
ax.set_title("quat1")
      
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat0)
ax.set_title("quat0")


"""
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quats_ori)
ax.set_title("quats_ori")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat2-quats_ori)
ax.set_title("quat2")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat1-quats_ori)
ax.set_title("quat1")
      
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat0-quats_ori)
ax.set_title("quat0")
"""

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.linalg.norm(quats_ori.astype(float),axis=1))
ax.plot(np.linalg.norm(quat2.astype(float),axis=1))
ax.plot(np.linalg.norm(quat1.astype(float),axis=1))
ax.plot(np.linalg.norm(quat0.astype(float),axis=1))

ax.set_title("norms_ori")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.linalg.norm(quat1.astype(float),axis=1))
ax.set_title("norms_quat1")

ground_truth = pd.read_csv('raw_data/harbor_colmap_traj_sequence_01.txt', sep=r'\s+', header=None)
ori0 = ground_truth.iloc[:,4:8].values
n_ori0 = np.array([np.linalg.norm(o) for o in ori0])

q0 = np.zeros((N,3))
q1 = np.zeros((N,3))
q2 = np.zeros((N,3))


for i in range(N):
    q0[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion0[i,:])))[1:4]
    q1[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion1[i,:])))[1:4]
    q2[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(newset.neworient[i,:])))[1:4]
    #q2[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion2[i,:])))[1:4]
q0z = np.zeros((N,3))
q1z = np.zeros((N,3))
q2z = np.zeros((N,3))


for i in range(N):
    q0z[i,:] = np.array(quat_rot([0,0,1,0],quat_inv(quaternion0[i,:])))[1:4]
    q1z[i,:] = np.array(quat_rot([0,0,1,0],quat_inv(quaternion1[i,:])))[1:4]
    #q2z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(newset.neworient[i,:])))[1:4]
    q2z[i,:] = np.array(quat_rot([0,0,1,0],quat_inv(quaternion2[i,:])))[1:4]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(np.arctan2(-q2z[:,1],q2z[:,2]))
ax.plot(np.arctan2(-q1z[:,1],q1z[:,2]))
ax.plot(np.arctan2(-q0z[:,1],q0z[:,2]))
ax.legend(['revmekf','mekf','gyro'])
ax.set_title('roll q2 q1 q0')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(-q2z[:,0],np.sqrt(q2z[:,2]**2+q2z[:,1]**2)))
ax.plot(np.arctan2(-q1z[:,0],np.sqrt(q1z[:,2]**2+q1z[:,1]**2)))
ax.plot(np.arctan2(-q0z[:,0],np.sqrt(q0z[:,2]**2+q0z[:,1]**2)))
ax.legend(['revmekf','mekf','gyro'])
ax.set_title('pitch q2 q1 q0')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(q2[:,1],q2[:,0]))
ax.plot(np.arctan2(q1[:,1],q1[:,0]))
ax.plot(np.arctan2(q0[:,1],q0[:,0]))
ax.legend(['revmekf','mekf','gyro'])
ax.set_title("q0 q1 heading")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q0)
ax.set_title("q0")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q1)
ax.set_title("q1")

for i in range(N):
    q0[i,:] = np.array(quat_rot([0,*newset.mag0],quat_inv(quaternion0[i,:])))[1:4]
    q1[i,:] = np.array(quat_rot([0,*newset.mag0],quat_inv(quaternion1[i,:])))[1:4]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q0)
ax.set_title("q0")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q1)
ax.set_title("q1")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.mag)
ax.set_title("newset mag")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q0z)
ax.set_title("q0z")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q1z)
ax.set_title("q1z")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)
ax.set_title("newset acc")

dquats_ori = np.zeros((N,3))
dquats0 = np.zeros((N,3))
dquats1 = np.zeros((N,3))
per = 1
for i in range(0,N-per,per):
    print(i)
    #lq =(np.array(quat_mult(quat_inv(RotToQuat(R.from_quat(q_ori[i+1,:]).as_matrix())),(RotToQuat(R.from_quat(q_ori[i,:]).as_matrix()))))).astype(float)
    lq =(np.array(quat_mult(quat_inv(ref[i+per]),(ref[i])))).astype(float)
    lq0 =(np.array(quat_mult(quat_inv(quaternion0[i+per]),(quaternion0[i])))).astype(float)
    lq1 =(np.array(quat_mult(quat_inv(quaternion1[i+per]),(quaternion1[i])))).astype(float)
    print(lq,lq0,lq1)
    dquats_ori[i,:] = np.array(log_q(lq/np.linalg.norm(lq)))
    dquats0[i,:] = np.array(log_q(lq0/np.linalg.norm(lq0)))
    dquats1[i,:] = np.array(log_q(lq1/np.linalg.norm(lq1)))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(dquats_ori)
ax.set_title("dquats_ori")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(dquats0)
ax.set_title("dquats0")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(dquats1)
ax.set_title("dquats1")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.gyro[:,:]/newset.freq)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.linalg.norm(quat2.astype(float)-quats_ori.astype(float),axis=1))
ax.plot(np.linalg.norm(quat1.astype(float)-quats_ori.astype(float),axis=1))
ax.plot(np.linalg.norm(quat0.astype(float)-quats_ori.astype(float),axis=1))
ax.set_title("dquats")