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

from kfilterdata2 import KFilterDataFile


mp.dps = 40
#mp.prec = 40

import sys,os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/examples')[0])


from scipy.spatial.transform import Rotation
from solver_kalman import SolverFilterPlan
from signal_process import compute_normals

g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)

data_file = 'selected_data_vehicle1.csv'
data=pd.read_csv(data_file)

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
n_end=n_start +600
cols = np.array([0,1,2,3,10,11,12,19,20,21])
df = data.values[n_start:n_end,cols]

acc = np.copy(df[:,1:4])
mag = np.copy(df[:,7:10])

fs = 50
sos = butter(2, 24, fs=fs, output='sos')
accs = np.copy(df[:,1:7])

df[:,4:7]=df[:,4:7]*np.pi/180

time= np.array(df[:,0],dtype=mpf)#/10**9
df[:,0]*=10**9



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


gravity = [0,0,np.mean(np.linalg.norm(acc_smooth[:150,:],axis=1))]

normals = compute_normals(N,acc_smooth,gravity,df[:,7:10])
y = normals
y_smooth0 = savgol_filter(y[:,0], 500, 2)
y_smooth1 = savgol_filter(y[:,1], 500, 2)
y_smooth2 = savgol_filter(y[:,2], 500, 2)
y_smooth = np.vstack((y_smooth0,y_smooth1,y_smooth2)).T
normals = np.copy(y_smooth)


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

q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(0), 10**(0)
normal = newset.normal


gravity = newset.gravity



proj_func = correct_proj2
proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv1 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True,neural=True)#,grav=newset.grav)
Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True,manual=True)#,grav=newset.grav)

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
correction_not_applied = np.zeros(N)
labels = np.empty(N, dtype=str)
angle_applied = np.zeros(N)
array_t0 = np.zeros(N)
array_t2 = np.zeros(N)
array_t3 = np.zeros(N)
array_t4 = np.zeros(N)

array_et0 = np.zeros(N)
array_et2 = np.zeros(N)
array_et3 = np.zeros(N)
array_et4 = np.zeros(N)

for i in range(0,N-1,1):
    
    nn+=1
    normal = normals[i+1,:]
    
    std_acc_z =0
    if i<N-1-40 and i> 40:
        std_acc_z = np.std(newset.acc[i-20:i+20,2])
    std_acc_zs[i+1] = std_acc_z 
    #print(std_acc_z)
    print("iteration",i)
    newset.acc[i+1,2] = s_acc_z[i+1]
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal,std_acc_z=std_acc_z)
    correction_applied[i+1] = Solv2.KFilter.corrected
    correction_not_applied[i+1] = Solv2.KFilter.not_corrected
    labels[i+1] = Solv2.KFilter.label
    angle_applied[i+1] =angle_applied[i]+Solv2.KFilter.angle
    
    array_t0[i+1] = Solv2.KFilter.t0
    array_t2[i+1] = Solv2.KFilter.t2
    array_t3[i+1] = Solv2.KFilter.t3
    array_t4[i+1] = Solv2.KFilter.t4
    
    array_et0[i+1] = Solv2.KFilter.et0
    array_et2[i+1] = Solv2.KFilter.et2
    array_et3[i+1] = Solv2.KFilter.et3
    array_et4[i+1] = Solv2.KFilter.et4

    if i%10 ==0 and i>0:
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot([np.linalg.norm(Solv0.position[j,:]) for j in range(i+1)])
        ax.plot([np.linalg.norm(Solv1.position[j,:]) for j in range(i+1)])
        ax.plot([np.linalg.norm(Solv2.position[j,:]) for j in range(i+1)])        
        ax.plot([np.linalg.norm(coords[j,:]) for j in range(i+1)])
        ax.plot(np.argwhere(correction_applied).flatten()-1, [np.linalg.norm(Solv2.position[j,:])  for j in np.argwhere(correction_applied).flatten()-1],'.',**dict(markersize=10))
        ax.plot(np.argwhere(correction_not_applied).flatten()-1, [np.linalg.norm(Solv2.position[j,:])  for j in np.argwhere(correction_not_applied).flatten()-1],'.',**dict(markersize=10))
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
 



size  =n_end-n_start

#size = n_start+N


time0=time-time[0]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],[np.linalg.norm(Solv0.position[j,:]) for j in range(size-1)])
ax.plot(time0[:size-1],[np.linalg.norm(Solv1.position[j,:]) for j in range(size-1)])
ax.plot(time0[:size-1],[np.linalg.norm(Solv2.position[j,:]) for j in range(size-1)])
ax.plot(time0[:size-1],[np.linalg.norm(coords[j,:]) for j in range(size-1)])
ax.plot(time0[np.argwhere(correction_applied).flatten()], [np.linalg.norm(Solv2.position[j,:]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.legend(['Integration of Gyroscope','MEKF','Rev-MEKF','GPS','Corrections applied by Rev-MEKF'])
ax.set_title('Distance computed in meters')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot([np.arctan2(coords[i,1],coords[i,0]) for i in range(len(coords))])
ax.set_title('gps')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(angle_applied)
ax.set_title('Cumulated corrected angles in radians')

acc_earth = np.array([np.array(quat_rot([0,*newset.acc[i,:]], quaternion2[i,:]))[1:4] for i in range(size-1)])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],np.linalg.norm(acc_earth-Solv2.KFilter.gravity,axis=1))
ax.set_xlabel('Seconds')
ax.set_ylabel('m.s^(-2)')
ax.set_title('Norm of external acceleration')

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

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],[np.linalg.norm(newset.acc[j,:]) for j in range(size-1)])
ax.plot(time0[np.argwhere(correction_applied).flatten()], [np.linalg.norm(newset.acc[j,:]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.legend(['Acc','Corrections applied by Rev-MEKF'])
ax.set_title('Corrected places on acc')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],[newset.acc[j,:] for j in range(size-1)])
ax.plot(time0[np.argwhere(correction_applied).flatten()], [newset.acc[j,:] for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.legend(['Acc','Corrections applied by Rev-MEKF'])
ax.set_title('Corrected places on acc')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],[newset.mag[j,:] for j in range(size-1)])
ax.plot(time0[np.argwhere(correction_applied).flatten()], [newset.mag[j,:] for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.legend(['Mag','Corrections applied by Rev-MEKF'])
ax.set_title('Corrected places on mag')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],[newset.gyro[j,:] for j in range(size-1)])
ax.plot(time0[np.argwhere(correction_applied).flatten()], [newset.gyro[j,:] for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.legend(['Gyro','Corrections applied by Rev-MEKF'])
ax.set_title('Corrected places on gyro')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],acc_smooth[:size-1,:2])
ax.plot(time0[np.argwhere(correction_applied).flatten()], [acc_smooth[j,:2] for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))
ax.set_title('Acc smoothed')


coords1 = np.zeros(coords.shape)
coords1[:,0] = (coords[:,0]+coords[:,1])/np.sqrt(2)
coords1[:,1] = (coords[:,0]-coords[:,1])/np.sqrt(2)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.array(coords1))
ax.set_title('gps')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0)
ax.plot(np.array(coords1))
ax.set_title('Position from Gyro Integration')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position1)
ax.plot(np.array(coords1))
ax.set_title('Position from MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2)
ax.plot(np.array(coords1))
ax.set_title('Position from Rev-MEKF')

q0 = np.zeros((N,3))
q1 = np.zeros((N,3))
q2 = np.zeros((N,3))

for i in range(N):
    q0[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion0[i,:])))[1:4]
    q1[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion1[i,:])))[1:4]
    q2[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(newset.neworient[i,:])))[1:4]
    
    
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


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.new_orient()[:,[0,1,3]])
ax.set_title('neworient')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.mag)
ax.set_title('mag')


dacc_smooth = np.diff(acc_smooth,axis=0)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],dacc_smooth[:size-1,:2])
ax.plot(time0[np.argwhere(correction_applied).flatten()-1], [dacc_smooth[j,:2] for j in np.argwhere(correction_applied).flatten()-1],'.',**dict(markersize=10))
ax.set_title('dAcc smoothed')

#correction_applied[30]=1
p0 = np.argwhere(correction_applied).flatten()[0]
p_start = p0-10
p_end = p0+10
    
rows = []


window = 20

for p1 in range(window,N,1):
    
    p0=p1
    if p1 in np.argwhere(correction_applied).flatten() or p1 in np.argwhere(correction_not_applied).flatten():
        
        p_start = max(0, p0 - window)
        p_end   = min(len(time0), p0 + window)
        indices = list(range(p_start, p0+1))  # indices du voisinage
        
        
        
    
        row = {
            "sample": int(p0)+n_start,
            "time": float(time0[p0])+time[0],
            "correction_applied": p0 in np.argwhere(correction_applied).flatten(),
        }
    
        for k, j in enumerate(indices):
            row[f"t0_{k}"] = float(array_t0[j])
            row[f"t2_{k}"] = float(array_t2[j])
            row[f"t3_{k}"] = float(array_t3[j])
            row[f"t4_{k}"] = float(array_t4[j])
            
            row[f"et0_{k}"] = float(array_et0[j])
            row[f"et2_{k}"] = float(array_et2[j])
            row[f"et3_{k}"] = float(array_et3[j])
            row[f"et4_{k}"] = float(array_et4[j])
        for k,j in enumerate(indices):
            row["label"] = str(labels[j])
        for k, j in enumerate(indices):
            row[f"acc_x_{k}"] = float(newset.acc[j,0])
        for k, j in enumerate(indices):
            row[f"acc_y_{k}"] = float(newset.acc[j,1])
        for k, j in enumerate(indices):
            row[f"acc_z_{k}"] = float(newset.acc[j,2])
    
        # Gyro
        for k, j in enumerate(indices):
            row[f"gyro_x_{k}"] = float(newset.gyro[j,0])
        for k, j in enumerate(indices):
            row[f"gyro_y_{k}"] = float(newset.gyro[j,1])
        for k, j in enumerate(indices):
            row[f"gyro_z_{k}"] = float(newset.gyro[j,2])
    
        # Mag
        for k, j in enumerate(indices):
            row[f"mag_x_{k}"] = float(newset.mag[j,0])
        for k, j in enumerate(indices):
            row[f"mag_y_{k}"] = float(newset.mag[j,1])
        for k, j in enumerate(indices):
            row[f"mag_z_{k}"] = float(newset.mag[j,2])
            
            
        for k, j in enumerate(indices):
            row[f"normal_x_{k}"] = float(normals[j,0])
        for k, j in enumerate(indices):
            row[f"normal_y_{k}"] = float(normals[j,1])
        for k, j in enumerate(indices):
            row[f"normal_z_{k}"] = float(normals[j,2])
    
    
        rows.append(row)
df = pd.DataFrame(rows)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"corrections_windows_angles_{timestamp}"+ str(n_start)+".csv", index=False)