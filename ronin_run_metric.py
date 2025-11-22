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
from ronin2 import *

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


imu_csv=pd.read_csv('oxiord/running/data1/raw/imu1.csv')


ground_truth = pd.read_csv('oxiord/running/data1/raw/vi1.csv')


# Extraire les colonnes 1 à 4 (attention, Python est 0-indexé → colonnes 1:5)
gps = ground_truth.values[:, 1:4]
ori = ground_truth.values[:,5:9]
ori = np.array([o/np.linalg.norm(o) for o in ori])
ground_truth['timestamp'] = ground_truth.values[:, 0]/10**(9)
gps_v = gps[:,[0,1]]
grav=9.80

acc_v = (-imu_csv.values[:,7:10]-imu_csv.values[:,10:13])*grav#-imu_csv.values[:,10:13])*grav

t_acc = imu_csv.values[:,0]#/10**(9)
mag_v = imu_csv.values[:,13:16]
mag_v0 = np.copy(mag_v)
t_mag = imu_csv.values[:,0]#/10**(9)
"""mag_v[:,0] = mag_v0[:,1]
mag_v[:,1] = mag_v0[:,0]
mag_v[:,2] = -mag_v0[:,2]"""





gyro_v = imu_csv.values[:,4:7]
t_gyro= imu_csv.values[:,0]#/10**(9)


acc_df = pd.DataFrame({
    'timestamp': t_acc,
    'acc_x': acc_v[:,0],
    'acc_y': acc_v[:,1],
    'acc_z': acc_v[:,2]
}).sort_values('timestamp')

gyro_df = pd.DataFrame({
    'timestamp': t_gyro,
    'gyro_x': gyro_v[:,0],
    'gyro_y': gyro_v[:,1],
    'gyro_z': gyro_v[:,2]
}).sort_values('timestamp')

mag_df = pd.DataFrame({
    'timestamp': t_mag,
    'mag_x': mag_v[:,0],
    'mag_y': mag_v[:,1],
    'mag_z': mag_v[:,2]
}).sort_values('timestamp')
"""'ori_w': ori[:,0],
'ori_x': ori[:,1],
'ori_y': ori[:,2],
'ori_z': ori[:,3],"""
gps_df = pd.DataFrame({
    'timestamp': ground_truth['timestamp'],#*10**9,  # ou ton propre vecteur GPS timestamps
    'gps_x': gps_v[:,0],
    'gps_y': gps_v[:,1],
    'ori_x': ori[:,0],
    'ori_y': ori[:,1],
    'ori_z': ori[:,2],
    'ori_w': ori[:,3],
    #'speed': gps[:,2] if gps.shape[1] > 2 else np.nan
}).sort_values('timestamp')

mpu = pd.merge_asof(acc_df, gyro_df, on='timestamp', direction='nearest')
mpu = pd.merge_asof(mpu, mag_df, on='timestamp', direction='nearest')
mpu = pd.merge_asof(mpu, gps_df, on='timestamp', direction='nearest')


mpu = mpu.dropna(how="all")
mpu = mpu.drop_duplicates(subset='timestamp', keep='first')

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
n_end=n_start +25000
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
new_mags = mpu[['mag_x','mag_y','mag_z']].rename(columns={'mag_x': 'x', 'mag_y': 'y', 'mag_z': 'z'})
filtered_mags = filter_outlier(new_mags, 0.01, 0.02)

offset_x, offset_y, offset_z, filtered_offset_mags = apply_offset_correction(filtered_mags)
scale_x, scale_y, scale_z, filtered_offset_scale_mags = apply_scale_correction(filtered_offset_mags)

N = len(mags[:,0])
N0 = len(filtered_offset_scale_mags.values[:,0])
mX,mY,mZ = new_mags.values[:,0],new_mags.values[:,1],new_mags.values[:,2]#.reshape((N,1)),filtered_mags[:,1].reshape((N,1)),filtered_mags[:,2].reshape((N,1))
mX,mY,mZ = new_mags.values[0:5000:1,0],new_mags.values[0:5000:1,1],new_mags.values[0:5000:1,2]
#mX,mY,mZ = filtered_mags.values[:10000,0],filtered_mags.values[:10000,1],filtered_mags.values[:10000,2] #.reshape((N,1)),filtered_mags[:,1].reshape((N,1)),filtered_mags[:,2].reshape((N,1))
"""filtered_mags = filter_outlier(new_mags, 0.02, 0.05).values



N = len(mags[:,0])
mX,mY,mZ = filtered_mags[:,0],filtered_mags[:,1],filtered_mags[:,2]#.reshape((N,1)),filtered_mags[:,1].reshape((N,1)),filtered_mags[:,2].reshape((N,1))
"""
#Minv, b, Ainv = Calibrate_Mag(mX.reshape((N0,1)), mY.reshape((N0,1)), mZ.reshape((N0,1)))

def calib_mag(mX,mY,mZ):
    Q, n, d = fitEllipsoid(mX.flatten(), mY.flatten(), mZ.flatten())
    print(n)
    print("d",d)
    Qinv = np.linalg.inv(Q)
    b = -np.dot(Qinv, n)
    print(Qinv)
    print(np.dot(n.T, np.dot(Qinv, n)))
    print(np.real(1 / np.sqrt(np.dot(n.T, np.dot(Qinv, n)) - d)) )
    Ainv = np.real(1 / np.sqrt(np.dot(n.T, np.dot(Qinv, n)) - d) * linalg.sqrtm(Q))
    N0 = len(mX)
    return Calibrate_Mag(mX.reshape((N0,1)), mY.reshape((N0,1)), mZ.reshape((N0,1)))
    #return Qinv,b,Ainv
Q, n, d = fitEllipsoid(mX.flatten(), mY.flatten(), mZ.flatten())
Qinv = np.linalg.inv(Q)
b = -np.dot(Qinv, n)
Ainv = np.real(1 / np.sqrt(np.dot(n.T, np.dot(Qinv, n)) - d) * linalg.sqrtm(Q))
#Qinv,b,Ainv= calib_mag(mX, mY, mZ)
#Qinv,b,Ainv= calib_mag(filtered_offset_mags.values[:,0], filtered_offset_mags.values[:,1], filtered_offset_mags.values[:,2])
#Ainv = np.array([[ 0.05486116, -0.00011346, 0.00146397],
#                                  [-0.00011346,  0.05951535,  0.00731025],
#                                  [0.00146397,  0.00731025,  0.07483137]])
#b = np.array([-2.50627418, -11.4804655, -29.53963506]).transpose() 

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_mags['x'], new_mags['y'], new_mags['z'], s=5, color='r')
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
plt.show()

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
plt.show()

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_offset_mags['corrected_x'],filtered_offset_mags['corrected_y'],filtered_offset_mags['corrected_z'], s=5, color='r')
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
plt.show()

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_offset_scale_mags['scaled_corrected_x'],filtered_offset_scale_mags['scaled_corrected_y'],filtered_offset_scale_mags['scaled_corrected_z'], s=5, color='r')
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
plt.show()



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
plt.show()

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

g_bias = np.zeros(3)
bb =np.mean(df[:100,4:7],axis=0)

newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23,normals=normals,gravity=np.array([0,0,grav]))#,gravity=np.array([0,0,grav],dtype=mpf))#,start=np.array(q_ori[0,:],dtype=mpf))#,gravity=np.array([0,0,9.76],dtype=mpf))#,normal=np.array([0.1101,1,0])) 
N=newset.size
#N=len(df)
nn = N-1
g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)
angle = int(N/2)

orient = newset.orient
pos_earth = newset.pos_earth

q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(6), 10**(6)
normal = newset.normal


gravity = newset.gravity








         
proj_func = correct_proj2
proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv3 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)

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


q_ori = mpu[['ori_w','ori_x','ori_y','ori_z']].values[n_start:n_end,:]

xs = np.copy(q_ori)

for i in range(len(q_ori)):
    q_ori[i,:] = xs[i,:]/np.linalg.norm(xs[i,:])



qqq0 = np.array(quat_mult(ExpQua(np.array([0,0,(0)])),ExpQua(np.array([0,0,(np.pi/8-np.pi/64)]))))
#qqq0 = np.array(quat_mult(ExpQua(np.array([0,0,(0)])),ExpQua(np.array([0,0,(np.pi/2)]))))
qqq1 = np.array([mpf('0.9996484354327505351079027482754353367820997'),
       mpf('-0.01754938359241596422050686462013313924100325'),
       mpf('0.01403845962988671022803964339446865007225605'),
       mpf('-0.01406933984238009143674571198286066914950821')])


rotated_q_ori0= np.array([quat_mult(qqq0,q_ori[i,:]) for i in range(len(q_ori))])

inv_newset1 = [np.array(quat_mult(quat_inv(rotated_q_ori0[j,:]),newset.neworient[j,:])) for j in range(100)]

qqq1 = np.mean(inv_newset1,axis=0)#quat_mult(newset.quat_calib,quat_inv(rotated_q_ori0[0,:]))
qqq1 = qqq1/np.linalg.norm(qqq1)
rotated_q_ori = np.array([quat_mult(rotated_q_ori0[i,:],qqq1) for i in range(len(q_ori))])
ref = rotated_q_ori


quats_ori = np.array([log_q(np.array(rotated_q_ori[i,:])) for i in range(len(q_ori))])
quat_accmag = np.array([log_q(np.array(newset.neworient[i,:])) for i in range(len(q_ori))])


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat_accmag)
ax.set_title("quataccmag")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quats_ori)
ax.set_title("quats_ori")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat_accmag-quats_ori)
ax.set_title("quataccmag")




fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(ref)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.neworient)

plt.show()

normals = np.zeros((N,3))
for i in range(0,N,1):
    normals[i] = np.array(quat_rot([0,1,0,0],ref[i,:]))[1:4]
    normals[i] = normals[i]/np.linalg.norm(normals[i])
    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(normals)
ax.set_title("normals")
#qq0 = quat_ntom(normals[4000,:], np.array([1,0,0]))
#normals = np.array([np.array(quat_rot([0,*normals[i,:]],qq0))[1:4] for i in range(len(normals))])
#gyr_gps = mpu[['w_x','w_y','w_z']].to_numpy()[n_start:n_end,:]

for i in range(0,N-1,1):
    
    nn+=1
    normal = normals[i+1,:]
    
    std_acc_z =0
    if i<N-1-40 and i> 40:
        std_acc_z = np.std(newset.acc[i-20:i+20,2])
    std_acc_zs[i+1] = std_acc_z 
    #print(std_acc_z)
    print("iteration",i)
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    #Solv3.update(time[i+1], gyr_gps[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    #Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal,std_acc_z=std_acc_z)
    correction_applied[i] = Solv2.KFilter.corrected
    angle_applied[i+1] =angle_applied[i]+Solv2.KFilter.angle

    
    if i%500 ==0 and i>0:
        """quat2 = np.array([log_q(np.array(quat_mult(Solv2.quaternion[j,:],quat_inv(Solv2.quaternion[0,:])))) for j in range(i+1)])
        quat1 = np.array([log_q(np.array(quat_mult(Solv1.quaternion[j,:],quat_inv(Solv1.quaternion[0,:])))) for j in range(i+1)])
        quat0 = np.array([log_q(np.array(quat_mult(Solv0.quaternion[j,:],quat_inv(Solv0.quaternion[0,:])))) for j in range(i+1)])
        """
        indices = np.argwhere(correction_applied).flatten()
        normn = [np.linalg.norm(log_q(np.array(quat_mult(Solv2.quaternion[j,:],quat_inv(ref[j,:]))))) for j in range(1,i+1)]
        fig, axs = plt.subplots(3, 2, figsize=(10, 8))  # 2x2 sous-graphiques
        axs = axs.flatten()  # pour accéder facilement via axs[k]
        ax = axs[0]
        ax.plot([np.linalg.norm(log_q(np.array(quat_mult(Solv0.quaternion[j,:],quat_inv(ref[j,:]))))) for j in range(1,i+1)], label='Solv0')
        ax.plot([np.linalg.norm(log_q(np.array(quat_mult(Solv1.quaternion[j,:],quat_inv(ref[j,:]))))) for j in range(1,i+1)], label='Solv1')
        #ax.plot([np.linalg.norm(log_q(np.array(quat_mult(Solv2.quaternion[j,:],quat_inv(ref[j,:]))))) for j in range(0,i+1)], label='Solv1')
        #ax.plot([quat2[j, k] for j in range(i+1)], label='Solv2')
        ax.plot(normn, label='Solv2')


        #ax.plot([quats_ori[j, k] for j in range(i+1)], label='ref')
        
        # points de correction
        
        if len(indices>0):
            ax.plot(indices, [normn[j-1] for j in indices], '.', markersize=10, label='correction')
    
        ax.set_title('Log dQuaternion')
        ax.legend()
        ax.grid(True)
        
        ax = axs[1]
        ax.plot([((np.array((Solv0.quaternion[j,:])))) for j in range(1,i+1)], label='Solv0')
        #ax.plot([quat2[j, k] for j in range(i+1)], label='Solv2')

        #ax.plot([quats_ori[j, k] for j in range(i+1)], label='ref')
        
        # points de correction
        #indices = np.argwhere(correction_applied).flatten()
        #ax.plot(indices, [quat2[j, k] for j in indices], '.', markersize=10, label='correction')
    
        ax.set_title('d normLogQuaternion')
        ax.legend()
        ax.grid(True)
        
        ax = axs[2]
        ax.plot([((np.array((Solv1.quaternion[j,:])))) for j in range(1,i+1)], label='Solv1')
        #ax.plot([quat2[j, k] for j in range(i+1)], label='Solv2')

        #ax.plot([quats_ori[j, k] for j in range(i+1)], label='ref')
        
        # points de correction
        #indices = np.argwhere(correction_applied).flatten()
        #ax.plot(indices, [quat2[j, k] for j in indices], '.', markersize=10, label='correction')
    
        ax.set_title('d normLogQuaternion')
        ax.legend()
        ax.grid(True)
        
        
        ax = axs[3]
        #ax.plot([((np.array((ref[j,:])))) for j in range(1,i+1)], label='Solv1')
        ax.plot([((np.array((Solv2.quaternion[j,:])))) for j in range(1,i+1)], label='Solv2')

        #ax.plot([quat2[j, k] for j in range(i+1)], label='Solv2')

        #ax.plot([quats_ori[j, k] for j in range(i+1)], label='ref')
        
        # points de correction
        if len(indices>0):
            ax.plot(indices, [np.array((Solv2.quaternion[j-1,:])) for j in indices], '.', markersize=10, label='correction')
    
        ax.set_title('d normLogQuaternion')
        ax.legend()
        ax.grid(True)
        ax = axs[4]
        ax.plot([((np.array((ref[j,:])))) for j in range(1,i+1)], label='ref')
        ax.set_title('d normLogQuaternion')
        ax.legend()
        ax.grid(True)
        
        
        plt.tight_layout()
        plt.show()
        
compare = np.zeros((N,4),dtype=float)
compare2 = np.zeros((N,4),dtype=float)
quaternion0 = Solv0.quaternion[:N,:]    
quaternion1 = Solv1.quaternion[:N,:]
quaternion2 = Solv2.quaternion[:N,:]
quaternion3 = Solv3.quaternion[:N,:]
position0 = Solv0.position[:N,:]
position1 = Solv1.position[:N,:]
position2 = Solv2.position[:N,:]
gravity_r = Solv2.gravity_r[:N,:]
metric0 = np.zeros(N,dtype=mpf)
metric1 = np.zeros(N,dtype=mpf)
metric2 = np.zeros(N,dtype=mpf)

time0=time-time[0]
size  =n_end-n_start
size=N
acc_earth = np.array([np.array(quat_rot([0,*newset.acc[i,:]], quaternion2[i,:]))[1:4] for i in range(size-1)])

ground_truth = np.zeros((N,4))
q_mean = np.mean(newset.neworient,axis=0)
q_mean = q_mean/np.linalg.norm(q_mean)
ground_truth= ref

for i in range(1,N):
    metric0[i] = metric0[i-1] + np.abs(1-quat_mult(quaternion0[i,:],quat_inv(ground_truth[i,:]))[0])
    metric1[i] = metric1[i-1] + np.abs(1-quat_mult(quaternion1[i,:],quat_inv(ground_truth[i,:]))[0])
    metric2[i] = metric2[i-1] + np.abs(1-quat_mult(quaternion2[i,:],quat_inv(ground_truth[i,:]))[0])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],metric0-metric1)
#ax.plot(time0[np.argwhere(correction_applied).flatten()], [np.linalg.norm((metric1-metric2)[j]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

#ax.plot(metric2)
ax.legend(['Lambda(X_Gyro,T) - Lambda(X_MEKF,T)','Correction applied'])
#plt.yscale("log")
plt.xlabel('Seconds')
plt.ylabel('Cumulated error')
ax.set_title('Difference of the metric computed by Gyro Integration and MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],metric1-metric2)
ax.plot(time0[np.argwhere(correction_applied).flatten()], [((metric1-metric2)[j]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

#ax.plot(metric2)
ax.legend(['Lambda(X_MEKF,T) - Lambda(X_REVMEKF,T)','Correction applied'])
#plt.yscale("log")
plt.xlabel('Seconds')
plt.ylabel('Cumulated error')
ax.set_title('Difference of the metric computed by MEKF and Heuristical Rev-MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],metric0)
ax.plot(time0[:size],metric1)
ax.plot(time0[:size],metric2)
ax.legend(['Lambda(X_Gyro,T)','Lambda(X_MEKF,T)','Lambda(X_REVMEKF,T)'])
#plt.yscale("log")
plt.xlabel('Seconds')
plt.ylabel('Cumulated error')
ax.set_title('Metric computed by MEKF and Heuristical Rev-MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot([quat_mult(quaternion0[i,:],quat_inv(ground_truth[i,:])) for i in range(len(ground_truth))])
ax.set_title("Quat_gyro*T^{-1}")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot([quat_mult(quaternion1[i,:],quat_inv(ground_truth[i,:])) for i in range(len(ground_truth))])
ax.set_title("Quat_MEKF*T^{-1}")
"""metric1[i] = metric1[i-1] + np.abs(1-quat_mult(quaternion1[i,:],quat_inv(ground_truth[i,:]))[0])
metric2[i] = metric2[i-1] + np.abs(1-quat_mult(quaternion2[i,:],quat_inv(ground_truth[i,:]))[0])"""

acc_ext = acc_earth-Solv2.KFilter.gravity
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],np.linalg.norm(acc_ext.astype(float),axis=1))
ax.plot(time0[np.argwhere(correction_applied).flatten()], [np.linalg.norm(acc_ext[j,:]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.set_xlabel('Seconds')
ax.set_ylabel('m.s^(-2)')
ax.set_title('Norm of external acceleration')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],np.linalg.norm(newset.acc.astype(float),axis=1))
ax.plot(time0[np.argwhere(correction_applied).flatten()], [np.linalg.norm(newset.acc.astype(float)[j,:]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.set_xlabel('Seconds')
ax.set_ylabel('m.s^(-2)')
ax.set_title('Norm of acceleration')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(gravity_r[1:size-1,2]*Solv2.KFilter.gravity[2])


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(time0[:size-1],newset.acc[1:size,2])
ax.plot(time0[:size-1],gravity_r[1:size,2]*Solv2.KFilter.gravity[2])
ax.set_xlabel('Seconds')
ax.set_ylabel('m.s^(-2)')
ax.legend(['Z coordinate of Accelerometer','Z coordinate of gravity vector returned by Rev-MEKF'])
ax.set_title('Comparison of z coordinates for different acceleration computed')

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
ax.plot(quat_accmag)
ax.set_title("quataccmag")
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

d_metric = metric1-metric2
dd_metric = np.diff(d_metric)
good = np.where(dd_metric>0)[0]
len(good)/N
interv = 40.0
a_paquets = np.arange(0,time0[size-1],interv)
paquets = np.zeros(len(a_paquets))
for i in range(0,len(a_paquets)):
    mask_total = (time0 >= a_paquets[i]) & (time0 < a_paquets[i] + interv)
    total_in_interval = mask_total.sum()
    print(total_in_interval)
    paquets[i] = len(np.where( (time0[good] < a_paquets[i] + interv) & (time0[good] >= a_paquets[i]))[0])/total_in_interval

    print("paquet ",i,paquets[i])