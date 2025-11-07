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

acc_gyro_csv=pd.read_csv('archeo/imu_sequence_1.csv')
mag_csv=pd.read_csv('archeo/mag_sequence_1.csv')
img=pd.read_csv('archeo/img_sequence_1.csv')

"""from pyubx2 import UBXReader
import csv
coords = []

with open("Geoloc_ds022023/S1/TEST_01/raw_measurement/gnss.ubx", "rb") as f:
    ubr = UBXReader(f)
    for (_, parsed_data) in ubr:
        if parsed_data.identity == "NAV-PVT":
            coords.append([
                parsed_data.iTOW/1e3,
                parsed_data.lat ,
                parsed_data.lon ,
                parsed_data.hMSL / 1000,
            ])

with open("Geoloc_ds022023/S1/TEST_01/raw_measurement/gnss.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["iTOW", "Latitude", "Longitude", "Altitude (m)"])
    writer.writerows(coords)
"""
ground_truth = pd.read_csv('archeo/new_archaeo_colmap_traj_sequence_01.txt', sep=r'\s+', header=None)

# Extraire les colonnes 1 à 4 (attention, Python est 0-indexé → colonnes 1:5)
gps = ground_truth.iloc[:, 1:4]
ori = ground_truth.iloc[:,4:8].values
#ori = np.array([o/np.linalg.norm(o) for o in ori])
ground_truth['timestamp'] = img[img.columns[0]].to_numpy()[ground_truth.iloc[:, 0].astype(int)]/10**(9)
gps_v = gps.values[:,[0,1]]

acc_v = acc_gyro_csv.values[:,4:7]
t_acc = acc_gyro_csv.values[:,0]/10**(9)
mag_v = mag_csv.values[:,1:4]#*1e6
mag_v0 = np.copy(mag_v)
t_mag = mag_csv.values[:,0]/10**(9)

mag_v[:,0] = mag_v0[:,2]
mag_v[:,1] = -mag_v0[:,1]
mag_v[:,2] = mag_v0[:,0]





gyro_v = acc_gyro_csv.values[:,1:4]
t_gyro= acc_gyro_csv.values[:,0]/10**(9)


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

"""mpu = pd.DataFrame(columns = ['timestamp','acceleration_x','acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z','mag_x','mag_y','mag_z','gps_x','gps_y','speed'])
mpu['acceleration_x']= 
mpu['acceleration_y']= 
mpu['acceleration_z']= 
mpu['gyro_x']= 
mpu['gyro_y']= 
mpu['gyro_z']=
mpu['mag_x']= 
mpu['mag_y']= 
mpu['mag_z']= 
mpu['gps_x'] =gps[:,0]
mpu['gps_y'] =gps[:,1]
mpu['timestamp'] = df_left.values[:,0]"""


"""t_common = np.arange(mpu['timestamp'].min(), mpu['timestamp'].max(), 5e6)  # pas de 10 ms

mpu = mpu.set_index('timestamp').reindex(t_common)
mpu.index.name = 'timestamp'"""

# Interpolation linéaire
#mpu = mpu.interpolate().reset_index()
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
n_end=n_start +10000
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

mags = np.array([mpu['mag_x'].values,mpu['mag_y'].values,mpu['mag_z'].values]).T
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
newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23,normals=normals,gravity=np.array([0,0,9.80],dtype=mpf))#,start=np.array(q_ori[0,:],dtype=mpf))#,gravity=np.array([0,0,9.76],dtype=mpf))#,normal=np.array([0.1101,1,0])) 
N=newset.size
#N=len(df)
nn = N-1
g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)
angle = int(N/2)

orient = newset.orient
pos_earth = newset.pos_earth

q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(-1), 10**(-1)
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
quats_ori[:,1] = xs[:,1]
quats_ori[:,0] = -xs[:,0]


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
    normal = np.array(quat_rot([0,0,1,0],ref[i+1,:]))[1:4]
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
    q0z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(quaternion0[i,:])))[1:4]
    q1z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(quaternion1[i,:])))[1:4]
    q2z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(newset.neworient[i,:])))[1:4]
    #q2z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(quaternion2[i,:])))[1:4]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.plot(np.arctan2(-q2z[:,1],q2z[:,2]))
ax.plot(np.arctan2(-q1z[:,1],q1z[:,2]))
ax.plot(np.arctan2(-q0z[:,1],q0z[:,2]))
ax.legend(['neworient','mekf','gyro'])
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