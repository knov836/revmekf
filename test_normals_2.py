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



#s_acc_z = acc_z
for i in range(0,N-1,1):
    
    nn+=1
    normal = np.array([0,0,1])
    std_acc_z =0
    if i<N-1-40 and i> 40:
        std_acc_z = np.std(newset.acc[i-20:i+20,2])
    std_acc_zs[i+1] = std_acc_z 
    #print(std_acc_z)
    print("iteration",i)
    #newset.acc[i+1,2] = s_acc_z[i+1]
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    #Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal,std_acc_z=std_acc_z)
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
        
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(s_acc_z)
ax.plot(acc_z)
ax.set_title('Evolution of s_acc_z')  
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(std_acc_zs)
ax.set_title('Evolution of std_acc_zs')
       
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(normals)
ax.set_title('Evolution of the normal')
 
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
nn=100
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0[:nn,0],position0[:nn,1])
#ax.plot(position1[:nn,0],position1[:nn,1])
ax.plot(coords1[:nn,0],coords1[:nn,1])
size  =n_end-n_start

#size = n_start+N

per = 500
speed = np.diff(coords[::per,:].astype(float),axis=0)
speed1 = np.diff(position1[::per,:].astype(float),axis=0)
speed0 = np.diff(position0[::per,:].astype(float),axis=0)
theta = np.arctan2(speed[:,1],speed[:,0])
theta1 = np.arctan2(speed1[:,1],speed1[:,0])
theta0 = np.arctan2(speed0[:,1],speed0[:,0])
theta1 = np.where(theta1  < 0, theta1 + 2 * np.pi, theta1)
theta = np.where(theta  < 0, theta + 2 * np.pi, theta)
theta0 = np.where(theta0  < 0, theta0 + 2 * np.pi, theta0)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(speed)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(speed1)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(theta0)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(theta)
ax.plot(theta0)
ax.plot(theta1)
q0 = np.zeros((N,3))
q1 = np.zeros((N,3))
q2 = np.zeros((N,3))


for i in range(N):
    q0[i,:] = np.array(quat_rot([0,1,0,0],(quaternion0[i,:])))[1:4]
    q1[i,:] = np.array(quat_rot([0,1,0,0],(quaternion1[i,:])))[1:4]
    q2[i,:] = np.array(quat_rot([0,1,0,0],(newset.neworient[i,:])))[1:4]
    
    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q0)
ax.set_title('q0')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q1)
ax.set_title('q1')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(q2)
ax.set_title('q2')

theta_q0 = np.arctan2(q0[::per,1],q0[::per,0])
theta_q0 = np.where(theta_q0 < 0, theta_q0 + 2 * np.pi, theta_q0 )


theta_q1 = np.arctan2(q1[::per,0],q1[::per,1])
theta_q1 = np.where(theta_q1 < 0, theta_q1 + 2 * np.pi, theta_q1 )


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(theta)
ax.plot(theta_q0)
ax.plot(theta_q1)


from scipy.ndimage import map_coordinates

x, y, z = newset.mag.T

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c=z, cmap='viridis', s=20)  # color by z just for visualization
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect((1, 1, 1))  # equal aspect ratio
plt.show()

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

# Create Rotation object
r = R.from_quat(quaternion1)

# Convert all to Euler angles (roll, pitch, yaw)
eulers = r.as_euler('xyz', degrees=True)

print("Euler angles (degrees):")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(eulers[:,1:])


r = R.from_quat(quaternion0)

# Convert all to Euler angles (roll, pitch, yaw)
eulers = r.as_euler('xyz', degrees=True)

print("Euler angles (degrees):")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(eulers[:,1:])

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
ax.plot(np.arctan2(q0[:,1],q0[:,0]))
ax.set_title("q0 heading")