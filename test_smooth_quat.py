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
accs = np.copy(df[:,1:4])

df[:,4:7]=df[:,4:7]*np.pi/180
gyro = np.copy(df[:,4:7])

time= np.array(df[:,0],dtype=mpf)#/10**9
#time = time-2*time[0]+time[1]
df[:,0]*=10**9
#time = time-2*time[0]+time[1]#
#df[:,7:10] = c_mag

#normal = np.mean(df[:100,7:10],axis=0)
df[:,1] = accs[:,1]
df[:,2] = accs[:,0]
df[:,3] = -accs[:,2]
df[:,4] = gyro[:,1]
df[:,5] = gyro[:,0]
df[:,6] = -gyro[:,2]


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

time0=time-time[0]


df = pd.read_csv('trajectory_mekf_20251017_215911.csv')
df = pd.read_csv('trajectory_mekf1_20251018_234213.csv')
#df = pd.read_csv('trajectory_mekf1_20251019_070050.csv')
df = pd.read_csv('trajectory_mekf_20251020_102553.csv')
df = pd.read_csv('trajectory_mekf1_20251023_011639.csv')

position1 = df[['px', 'py', 'pz']].to_numpy()      # shape (N, 3)
quaternion1 = df[['qw', 'qx', 'qy', 'qz']].to_numpy() 

#df = pd.read_csv('trajectory_heuristic2_20251018_122509.csv')

df = pd.read_csv('trajectory_heuristic2_20251018_234213.csv')
df = pd.read_csv('trajectory_heuristic2_20251019_001728.csv')
df = pd.read_csv('trajectory_heuristic2_20251019_070050.csv')
df = pd.read_csv('trajectory_heuristic2_20251019_180631.csv')
df = pd.read_csv('trajectory_heuristic2_20251023_011639.csv')
position2 = df[['px', 'py', 'pz']].to_numpy()      # shape (N, 3)
quaternion2 = df[['qw', 'qx', 'qy', 'qz']].to_numpy() 



df = pd.read_csv('trajectory_gyro_20251017_214529.csv')

position0 = df[['px', 'py', 'pz']].to_numpy()      # shape (N, 3)
quaternion0 = df[['qw', 'qx', 'qy', 'qz']].to_numpy() 

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion1)


window = 1000
q1w = savgol_filter(quaternion1[:,0], window , 2)
q1x = savgol_filter(quaternion1[:,1], window , 2)
q1y = savgol_filter(quaternion1[:,2], window , 2)
q1z = savgol_filter(quaternion1[:,3], window , 2)
quaternion1= np.vstack((q1w,q1x,q1y,q1z)).T


window = 1000
q2w = savgol_filter(quaternion2[:,0], window , 2)
q2x = savgol_filter(quaternion2[:,1], window , 2)
q2y = savgol_filter(quaternion2[:,2], window , 2)
q2z = savgol_filter(quaternion2[:,3], window , 2)
quaternion2= np.vstack((q2w,q2x,q2y,q2z)).T

pos0,speed0=quat_to_pos(time0,quaternion0,newset.acc,newset.gravity)

pos1,speed1=quat_to_pos(time0,quaternion1,newset.acc,newset.gravity)
pos2,speed2=quat_to_pos(time0,quaternion2,newset.acc,newset.gravity)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos1)
ax.plot(coords)
ax.set_title("pos1")
q0 = np.zeros((N,3))
q1 = np.zeros((N,3))
q2 = np.zeros((N,3))
q3 = np.zeros((N,3))
gq = np.zeros((N,3))
q0z = np.zeros((N,3))
q1z = np.zeros((N,3))
q2z = np.zeros((N,3))
gqz = np.zeros((N,3))

global_quatw = savgol_filter(newset.neworient[:,0], window , 2)
global_quatx = savgol_filter(newset.neworient[:,1], window , 2)
global_quaty = savgol_filter(newset.neworient[:,2], window , 2)
global_quatz = savgol_filter(newset.neworient[:,3], window , 2)
quat_accmag= np.vstack((global_quatw,global_quatx,global_quaty,global_quatz)).T
pos_accmag,speed_accmag=quat_to_pos(time0,quat_accmag,newset.acc,newset.gravity)


for i in range(N):
    q0[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion0[i,:])))[1:4]
    q1[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion1[i,:])))[1:4]
    q2[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion2[i,:])))[1:4]
    gq[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quat_accmag[i,:])))[1:4]

for i in range(N):
    q0z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(quaternion0[i,:])))[1:4]
    q1z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(quaternion1[i,:])))[1:4]
    q2z[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(quaternion2[i,:])))[1:4]
    gqz[i,:] = np.array(quat_rot([0,0,0,1],quat_inv(quat_accmag[i,:])))[1:4]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q1)
ax.set_title('q1')
heading0 = np.arctan2(q0[:,1],q0[:,0])
heading1 = np.arctan2(q1[:,1],q1[:,0])
heading2 = np.arctan2(q2[:,1],q2[:,0])
per = 500
speed = np.diff(coords[::per,:].astype(float),axis=0)
theta = np.arctan2(-speed[:,1],speed[:,0])


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(q0[:,1],q0[:,0]))

ax.plot(np.arctan2(q1[:,1],q1[:,0]))
ax.plot(range(0,len(q1)-500,500),-theta+theta[0]+heading1[0])

ax.set_title("q1 heading")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.array(coords[:,1]),np.array(coords[:,0]))
ax.plot(pos1[:,0],pos1[:,1])
ax.legend(['GPS','Position from MEKF'])
plt.axis('equal')
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(-np.array(coords[:,0]),-np.array(coords[:,1]))
ax.plot(pos0[:,0],pos0[:,1])
ax.legend(['GPS','Position from gyro'])
plt.axis('equal')
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion1)
ax.set_title("quat 1")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion0)
ax.set_title("quat 0")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos0)
ax.set_title("pos0")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(speed0)
ax.set_title("speed0")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos1)
ax.set_title("pos1")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(speed1)
ax.set_title("speed1")


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)
ax.set_title("acc")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(accs)
ax.set_title("acc")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q0z)
ax.set_title("q0z")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q1z)
ax.set_title("q1z")


from scipy.spatial.transform import Rotation as R

quaternion3 = np.zeros((len(q0),4))
quaternion4 = np.zeros((len(q0),4))

roll0 = (np.arctan2(q0z[:,1],q0z[:,2]))
pitch0 = (np.arctan2(-q0z[:,0],np.sqrt(q0z[:,2]**2+q0z[:,1]**2)))
heading0 = np.arctan2(q0[:,1],q0[:,0])

roll1 = (np.arctan2(q1z[:,1],q1z[:,2]))
pitch1 = (np.arctan2(-q1z[:,0],np.sqrt(q1z[:,2]**2+q1z[:,1]**2)))
heading1 = np.arctan2(q1[:,1],q1[:,0])

roll2 = (np.arctan2(q2z[:,1],q2z[:,2]))
pitch2 = (np.arctan2(-q2z[:,0],np.sqrt(q2z[:,2]**2+q2z[:,1]**2)))
heading2 = np.arctan2(q2[:,1],q2[:,0])

euler_angles0 = np.column_stack((heading0, pitch0,roll0))
rot0 = R.from_euler('ZYX', euler_angles0, degrees=False)
euler_angles1 = np.column_stack((heading1, pitch1,roll1))
rot1 = R.from_euler('ZYX', euler_angles1, degrees=False)
euler_angles2 = np.column_stack((heading2, pitch2,roll2))
rot2 = R.from_euler('ZYX', euler_angles2, degrees=False)
print(rot0.inv().as_matrix()[0,:]@[0,0,1])
print(q0z[0,:])

print(rot0.inv().as_matrix()[0,:]@[1,0,0])
print(q0[0,:])

euler_hybrid1 = np.column_stack((heading1, pitch1,roll0))
roth1 = R.from_euler('ZYX', euler_hybrid1, degrees=False)

euler_hybrid2 = np.column_stack((heading2, pitch2,roll0))
roth2 = R.from_euler('ZYX', euler_hybrid2, degrees=False)
for i in range(len(quaternion3)):
    nacc1 = roth1.inv().as_matrix()[i,:]@[0,0,1]
    nacc2 = roth2.inv().as_matrix()[i,:]@[0,0,1]
    quaternion3[i,:] = (RotToQuat(acc_mag_to_rotation(nacc1,q1[i,:])['R']))
    quaternion4[i,:] = (RotToQuat(acc_mag_to_rotation(nacc2,q2[i,:])['R']))

for i in range(N):
    q3[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion3[i,:])))[1:4]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(q0[:,1],q0[:,0]))
ax.set_title("theta0")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(q3[:,1],q3[:,0]))
ax.set_title("theta3")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion0)
ax.set_title("quat 0")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion3)
ax.set_title("quat 3")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion4)
ax.set_title("quat 4")

pos3,speed3=quat_to_pos(time0,quaternion3,newset.acc,newset.gravity)
pos4,speed4=quat_to_pos(time0,quaternion4,newset.acc,newset.gravity)




fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos3)
ax.set_title("pos3")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_accmag)
ax.set_title("accmag")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quat_accmag)
ax.set_title("quat_accmag")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion0)
ax.set_title("quaternion0")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion1)
ax.set_title("quaternion1")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.array(-coords[:,0]),-np.array(coords[:,1]))
ax.plot(pos0[:,0],pos0[:,1])
ax.plot(pos3[:,0],pos3[:,1])
ax.plot(pos4[:,0],pos4[:,1])
ax.legend(['GPS','Gyro','Heading/Pitch from MEKF and Roll from gyro','Heading/Pitch from Rev MEKF and Roll from gyro'])
plt.axis('equal')
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/MEKF')





fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

#ax.plot(newset.acc)

ax.plot(-np.array(coords[:,0]),-np.array(coords[:,1]))
ax.plot(pos_accmag[:,0],pos_accmag[:,1])
ax.plot(position0[:,0],position0[:,1])
ax.plot(position1[:,0],position1[:,1])
ax.plot(position2[:,0],position2[:,1])

ax.legend(['GPS','accmag','Position from Gyro integration','Position from MEKF','Position from RevMEKF'])
plt.axis('equal')
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/MEKF')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(gqz[:,1],-gqz[:,2]))
ax.plot(np.arctan2(q2z[:,1],-q2z[:,2]))
ax.plot(np.arctan2(q1z[:,1],-q1z[:,2]))
ax.plot(np.arctan2(q0z[:,1],-q0z[:,2]))
ax.legend(['acc-mag','revmekf','mekf','gyro'])
ax.set_title('roll q2 q1 q0')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(-gqz[:,0],np.sqrt(gqz[:,2]**2+gqz[:,1]**2)))
ax.plot(np.arctan2(-q2z[:,0],np.sqrt(q2z[:,2]**2+q2z[:,1]**2)))
ax.plot(np.arctan2(-q1z[:,0],np.sqrt(q1z[:,2]**2+q1z[:,1]**2)))
ax.plot(np.arctan2(-q0z[:,0],np.sqrt(q0z[:,2]**2+q0z[:,1]**2)))
ax.legend(['acc-mag','revmekf','mekf','gyro'])
ax.set_title('pitch q2 q1 q0')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(gq[:,1],gq[:,0]))
ax.plot(np.arctan2(q2[:,1],q2[:,0]))
ax.plot(np.arctan2(q1[:,1],q1[:,0]))
ax.plot(np.arctan2(q0[:,1],q0[:,0]))
ax.legend(['acc-mag','revmekf','mekf','gyro'])
ax.set_title("heading")

per=int(250*4)
speed = np.diff(-coords[::per,:].astype(float),axis=0)*newset.freq
theta = np.arctan2(speed[:,1],speed[:,0])
mag0 = newset.mag.astype(float)
mag = np.array([np.array(quat_rot([0,*m],ExpQua(np.array([0,0.033,0]))))[1:4] for m in mag0]).astype(float)
delta = theta[int(2000/per)]-np.arctan2(q1[2000,1],q1[2000,0])
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.arctan2(mag[:,1],mag[:,0]))
#ax.plot(np.arctan2(q2[:,1],q2[:,0]))
#ax.plot(np.arctan2(q1[:,1],q1[:,0]))
ax.plot(np.arctan2(q0[:,1],q0[:,0]))
ax.plot(range(per,len(q2)-per+1,per),theta[:]-delta)

