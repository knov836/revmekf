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

q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(5), 10**(5)
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


for i in range(0,N-1,1):
    
    nn+=1
    normal = np.array([0,0,1])
    newset.acc[i+1,2] = s_acc_z[i+1]
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
quaternion0 = Solv0.quaternion[:N,:]   
position0 = Solv0.position[:N,:]


"""normals = compute_normals(N,acc_smooth,gravity,df[:,7:10])


y = normals
y_smooth0 = savgol_filter(y[:,0], 500, 2)
y_smooth1 = savgol_filter(y[:,1], 500, 2)
y_smooth2 = savgol_filter(y[:,2], 500, 2)
y_smooth = np.vstack((y_smooth0,y_smooth1,y_smooth2)).T
normals = np.copy(y_smooth)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(y_smooth)
ax.set_title('Evolution of the normal')
normals = np.copy(y_smooth)"""
normals = np.array([quat_rot([0,0,0,1],quaternion0[i,:]) for i in range(N)])[:,1:4]


#s_acc_z = acc_z
for i in range(0,N-1,1):
    
    nn+=1
    normal = normals[i+1,:]
    #normal = np.array([0,0,1])
    std_acc_z =0
    if i<N-1-40 and i> 40:
        std_acc_z = np.std(newset.acc[i-20:i+20,2])
    std_acc_zs[i+1] = std_acc_z 
    #print(std_acc_z)
    print("iteration",i)
    newset.acc[i+1,2] = s_acc_z[i+1]
    #Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
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
 
quaternion1 = Solv1.quaternion[:N,:]
quaternion2 = Solv2.quaternion[:N,:]
position1 = Solv1.position[:N,:]
position2 = Solv2.position[:N,:]
gravity_r = Solv2.gravity_r[:N,:]


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
ax.plot(time0[np.argwhere(correction_applied).flatten()], [acc_smooth[j,:2] for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10),color='red')
ax.legend(['X axis','Y axis','Corrections applied'])
plt.xlabel('seconds')
plt.ylabel('meters per seconds^2')
ax.set_title('Acceleration smoothed')


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
q3 = np.zeros((N,3))

for i in range(N):
    q0[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion0[i,:])))[1:4]
    q1[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion1[i,:])))[1:4]
    q2[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(quaternion2[i,:])))[1:4]
    q3[i,:] = np.array(quat_rot([0,1,0,0],quat_inv(newset.neworient[i,:])))[1:4]
    
    
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
ax.plot(time0[np.argwhere(correction_applied).flatten()], [dacc_smooth[j,:2] for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))
ax.set_title('dAcc smoothed')


p0 = np.argwhere(correction_applied).flatten()[0]
p_start = p0-10
p_end = p0+10
    
rows = []


window = 20

for p1 in range(0,N,10):
    p_start = max(0, p0 - window)
    p_end   = min(len(time0), p0 + window)
    indices = list(range(p_start, p_end))  # indices du voisinage
    
    
    corr_indices = np.argwhere(correction_applied).flatten()

    candidates = corr_indices[corr_indices >= p1]
    
    if len(candidates) > 0 and (candidates.min() - p1) <= 10:
        p0 = candidates.min()
    else:
        p0 = p1

    row = {
        "sample": int(p0)+n_start,
        "time": float(time0[p0])+time[0],
        "correction_applied": p0 in np.argwhere(correction_applied).flatten(),
    }

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
df.to_csv(f"corrections_windows_{timestamp}"+ str(n_start)+".csv", index=False)

coords1 = coords


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0[:,:2])
ax.plot(np.array(coords1))
ax.legend(['pos0_x','pos0_y','gps_x,','gps_y'])
ax.set_title('Position from Gyro Integration')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position1[:,:2])
ax.plot(np.array(coords1))
ax.legend(['pos0_x','pos0_y','gps_x,','gps_y'])
ax.set_title('Position from MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2[:,:2])
ax.plot(np.array(coords1))
ax.legend(['pos0_x','pos0_y','gps_x,','gps_y'])
ax.set_title('Position from Old Rev-MEKF')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position1[:,0],position1[:,1])
ax.set_title('Position with MEKF')



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.array(coords1[:,1]),np.array(coords1[:,0]))
ax.plot(position0[:,0],position0[:,1])
ax.plot(position1[:,0],position1[:,1])
ax.plot(position2[:,0],position2[:,1])
ax.legend(['GPS','Position from Gyro integration','Position from MEKF','Position from Rev MEKF'])
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/Rev-MEKF')

alpha=np.pi/2+np.pi/8+np.pi/64+np.pi/128
coords1 = np.zeros(coords.shape)
coords1[:,0] = (np.cos(alpha)*coords[:,0]+np.sin(alpha)*coords[:,1])
coords1[:,1] = (np.sin(alpha)*coords[:,0]-np.cos(alpha)*coords[:,1])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
#ax.plot(newset.acc)

ax.plot(np.array(coords1[:,1]),np.array(coords1[:,0]))
ax.plot(position0[:,0],position0[:,1])
ax.plot(position1[:,0],position1[:,1])
ax.plot(position2[:,0],position2[:,1])
ax.legend(['GPS','Position from Gyro integration','Position from MEKF','Position from Rev-MEKF'])
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/MEKF')


def rotation_2d_from_a_to_b(a, b, eps=1e-12):
    """
    Return 2x2 rotation matrix R such that R @ a = b (for 2D vectors).
    a, b: array-like of shape (2,)
    eps: tolerance for zero-length vectors
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.linalg.norm(a) < eps or np.linalg.norm(b) < eps:
        raise ValueError("a and b must be non-zero 2D vectors")

    # normalize (ensures purely rotational map)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    dot = a.dot(b)
    det = a[0]*b[1] - a[1]*b[0]   # scalar "2D cross product"

    theta = np.arctan2(det, dot)

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return R
"""p0=  coords1[-1,:2]/np.linalg.norm(coords1[-1,:2])
p1=  position0[-1,:2]/np.linalg.norm(position0[-1,:2])
p2=  position1[-1,:2]/np.linalg.norm(position1[-1,:2])
R0 = rotation_2d_from_a_to_b(p1,p0)
R1 = rotation_2d_from_a_to_b(p2,p0)

rposition0 = position0[:,:2]@R0.T
rposition1 = position1[:,:2]@R1.T"""

"""fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.array(coords1[:,0]),np.array(coords1[:,1]))
ax.plot(rposition0[:,0],rposition0[:,1])
ax.plot(rposition1[:,0],rposition1[:,1])
ax.legend(['GPS','Position from Gyro integration','Position from MEKF'])
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/MEKF')
"""

import pandas as pd
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data = np.hstack((position2, quaternion2))
columns = ['px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(f"trajectory_heuristic2_{timestamp}.csv", index=False)



#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data = np.hstack((position1, quaternion1))
columns = ['px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']    

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(f"trajectory_mekf1_{timestamp}.csv", index=False)

print("Saved to trajectory.csv")
print(df.head())
# Charger le CSV
"""df = pd.read_csv('trajectory_mekf1_20251015_103619.csv')

position3 = df[['px', 'py', 'pz']].to_numpy()      # shape (N, 3)
quaternion3 = df[['qw', 'qx', 'qy', 'qz']].to_numpy() """
"""fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(aa)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0[:,0],position0[:,1])
ax.plot(position5[:,0],position5[:,1])
ax.set_aspect('equal', adjustable='box')"""

"""speed = np.diff(position0,axis=0)
#ax.plot(newset.acc)

ax.plot(np.array(coords1[:,1]),np.array(coords1[:,0]))
ax.plot(position0[:,0],position0[:,1])
ax.plot(position1[:,0],position1[:,1])
ax.plot(position3[:,0],position3[:,1])
ax.legend(['GPS','Position from Gyro integration','Position from MEKF','Position from Rev-MEKF'])
plt.xlabel('X axis in meters')
plt.ylabel('Y axis in meters')
ax.set_title('Projected position in 2D of GPS/Gyro Integration/MEKF')"""


heading0 = np.arctan2(q0[::per,1],q0[::per,0])
heading1 = np.arctan2(q1[::per,1],q1[::per,0])
heading2 = np.arctan2(q2[::per,1],q2[::per,0])
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(heading0)
ax.plot(heading1)
ax.plot(heading2)
ax.plot(-theta+theta[0]+heading1[0])

ax.set_title("q1 heading")