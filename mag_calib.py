import numpy as np
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numdifftools as nd 
from scipy import *
from mpmath import mp
from mpmath import mpf
import pandas as pd
import sys,os


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

#data_file = 'calib_mag.csv'
#data_file = 'imu_data_rosbag.csv'
#data_file = 'imu_data_sbg.csv'
data_file = 'imu_data_sbg_250624_2259.csv'
data_file  ='imu_data_sbg_250630_2340.csv'
data_file = 'imu_data_sbg_250701_1120.csv'
file_name = data_file.split('imu')[1]
#data_file = 'static_260625.csv'
#N = 10000
data=pd.read_csv(data_file)
#print(data.head())
df = data.values

mags = np.array(df[:,7:10],dtype=mpf)

N = len(mags[:,0])
mX,mY,mZ = mags[:,0].reshape((N,1)),mags[:,1].reshape((N,1)),mags[:,2].reshape((N,1))
Minv, b, Ainv = Calibrate_Mag(mX, mY, mZ)
print(df)
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

calibratedX = np.zeros(mX.shape)
calibratedY = np.zeros(mY.shape)
calibratedZ = np.zeros(mZ.shape)

c_mag = np.zeros(mags.shape)
totalError = 0
for i in range(len(mX)):
    h = np.array([[mX[i], mY[i], mZ[i]]]).T
    hHat = np.matmul(Ainv, h-b)
    hHat = hHat[:, :, 0]
    calibratedX[i] = hHat[0][0]
    calibratedY[i] = hHat[0][1]
    calibratedZ[i] = hHat[0][2]
    mag = np.dot(hHat.T, hHat)
    err = (mag[0][0] - 1)**2
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

# plot unit sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax2.plot_wireframe(x, y, z, rstride=10, cstride=10, alpha=0.5)
ax2.plot_surface(x, y, z, alpha=0.3, color='b')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(mags[:4000,:])



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(c_mag)
print(len(c_mag))
print(c_mag)

np.savetxt("mag"+file_name, c_mag, delimiter=",", fmt='%f')  # %d pour des entiers