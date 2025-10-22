import pandas as pd
import numpy as np
from scipy import *

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

acc_columns_x = [
    'acc_x_dashboard',#,
    'acc_x_above_suspension',#,
    'acc_x_below_suspension'
]
acc_columns_y = [
    'acc_y_dashboard',#,
    'acc_y_above_suspension',#,
    'acc_y_below_suspension'
]
acc_columns_z = [
    'acc_z_dashboard',#,
    'acc_z_above_suspension',#,
    'acc_z_below_suspension'
]
gyro_columns_x = [
    'gyro_x_dashboard',
    #'gyro_x_above_suspension'#,
    #'gyro_x_below_suspension'
]
gyro_columns_y = [
    'gyro_y_dashboard'
    #'gyro_y_above_suspension',
    #'gyro_y_below_suspension'
]
gyro_columns_z = [
    'gyro_z_dashboard',
    #'gyro_z_above_suspension'#,
    #'gyro_z_below_suspension'
]
mag_columns_x = [
    'mag_x_dashboard',
    #'mag_x_above_suspension'#,'mag_x_below_suspension'
]
mag_columns_y = [
    'mag_y_dashboard',
    #'mag_y_above_suspension'#,'mag_y_below_suspension'
]
mag_columns_z = [
    'mag_z_dashboard',
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

df_left=pd.read_csv('dataset_gps_mpu_left.csv')
df_right=pd.read_csv('dataset_gps_mpu_right.csv')

def absolute(columns, axis):
    sum_left = df_left[columns].sum(axis=1).div(axis)
    print((sum_left))
    sum_right = df_right[columns].sum(axis=1).div(axis)
    return pd.concat([sum_left, sum_right], axis=1).mean(axis=1)


gps = df_left.values[:,[-3,-2]]

mpu = pd.DataFrame(columns = ['timestamp','acceleration_x','acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z','mag_x','mag_y','mag_z','gps_x','gps_y','speed'])
mpu['acceleration_x']= absolute(acc_columns_x, acc_x)
mpu['acceleration_y']= absolute(acc_columns_y, acc_y)
mpu['acceleration_z']= absolute(acc_columns_z, acc_z)
mpu['gyro_x']= absolute(gyro_columns_x, gyro_x)
mpu['gyro_y']= absolute(gyro_columns_y, gyro_y)
mpu['gyro_z']= absolute(gyro_columns_z, gyro_z)
mpu['mag_x']= absolute(mag_columns_x, mag_x)
mpu['mag_y']= absolute(mag_columns_y, mag_y)
mpu['mag_z']= absolute(mag_columns_z, mag_z)
mpu['gps_x'] =gps[:,0]
mpu['gps_y'] =gps[:,1]
mpu['timestamp'] = df_left.values[:,0]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(mpu['mag_x'])
ax.plot(mpu['mag_y'])
ax.plot(mpu['mag_z'])
ax.set_title('Mag')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(mpu['gyro_x'])
ax.plot(mpu['gyro_y'])
ax.plot(mpu['gyro_z'])
ax.set_title('Gyro')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(mpu['acceleration_x'])
ax.plot(mpu['acceleration_y'])
ax.plot(mpu['acceleration_z'])
ax.set_title('Acc')


mag = np.array([mpu['mag_x'],mpu['mag_y'],mpu['mag_z']]).T

acc= np.array([mpu['acceleration_x'],mpu['acceleration_y'],mpu['acceleration_z']]).T
gyro= np.array([mpu['gyro_x'],mpu['gyro_y'],mpu['gyro_z']]).T
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(gyro[9000:19000,:])
ax.set_title('gyro')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(acc[9000:12000,:])
ax.set_title('acc')
"""fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(mag[9000:19000,:])
ax.set_title('Mag')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(mag[9000:19000,1],mag[9000:19000,0]))
"""
"""
from mpl_toolkits.mplot3d import Axes3D  # (not strictly needed in modern versions)

# Example: random data of size (N, 3)
N = 100
points = mag[9000:19000,:]

# Extract columns
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c='blue', marker='o')

# Optional: label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


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
mags = mag[:,:]
N = len(mags[:,0])
mX,mY,mZ = mags[:,0].reshape((N,1)),mags[:,1].reshape((N,1)),mags[:,2].reshape((N,1))
Minv, b, Ainv = Calibrate_Mag(mX, mY, mZ)
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




print("A_inv: ")
print(Ainv)
print()
print("b")
print(b)
print()

calibratedX = np.zeros(mag[9090:19090,0].shape)
calibratedY = np.zeros(mag[9090:19090,1].shape)
calibratedZ = np.zeros(mag[9090:19090,2].shape)

c_mag = np.zeros(mag[9090:19090,:].shape)
totalError = 0
for i in range(len(c_mag)):
    h = np.array([np.array([mag[9090:19090,0][i], mag[9090:19090,1][i], mag[9090:19090,2][i]]).flatten().reshape((3,1))])
    hHat = np.matmul(Ainv, h-b)
    hHat = hHat[:, :, 0]
    calibratedX[i] = hHat[0][0]
    calibratedY[i] = hHat[0][1]
    calibratedZ[i] = hHat[0][2]
    mag0 = np.dot(hHat.T, hHat)
    err = (mag0[0][0] - 1)**2
    totalError += err
print("Total Error: %f" % totalError)
c_mag[:,0] = calibratedX.flatten()
c_mag[:,1] = calibratedY.flatten()
c_mag[:,2] = calibratedZ.flatten()

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, projection='3d')

ax3.scatter(mX, mY, mZ, s=5, color='r')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax3.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)
ax3.plot_surface(x, y, z, alpha=0.3, color='b')


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')

ax2.scatter(calibratedX, calibratedY, calibratedZ, s=1, color='r')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(c_mag)
ax.set_title('c Mag')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(c_mag[:,1],c_mag[:,0]))
"""