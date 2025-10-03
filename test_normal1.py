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


g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)


data_file = 'imu_data5.csv'
data_file = 'calibration_data_250523.csv'
data_file = 'calibration_020625.csv'
data_file = 'calibration_030625.csv'
data_file = 'imu_data_sbg_250624_2055.csv'

"""data_file = 'imu_data_sbg.csv'

#N = 10000
data=pd.read_csv(data_file)
#print(data.head())
df = data.values
time= np.array(df[:,1],dtype=mpf)/10**9
"""
#data_file = 'imu_data_sbg_250624_2259.csv'
data_file = 'imu_data_sbg_250624_2055.csv'

#static
#data_file= 'imu_data_sbg_250701_1120.csv'
data_file= 'imu_data_sbg_250701_1120.csv'
data_file = 'imu_data_sbg_static_120925.csv'
#moving
#data_file= 'imu_data_sbg_250701_1344.csv'
#data_file= 'imu_data_sbg_250701_1628.csv'
data_file = 'imu_data_sbg_plane_09121118.csv'
data_file = 'imu_data_sbg_plane_09121136.csv'
data_file = 'imu_data_sbg_plane_09121245.csv'
data_file = 'imu_data_sbg_plane_09121311.csv'
#slope
#data_file= 'imu_data_sbg_250701_1710.csv'

#data_file = 'odo_data.csv'

#dataset
#data_file = 'dataset_gps_mpu_left.csv'
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
n_end=n_start +2000
cols = np.array([0,1,2,3,10,11,12,19,20,21])
df = data.values[n_start:n_end,cols]

acc = np.copy(df[:,1:4])
mag = np.copy(df[:,7:10])

fs = 50
sos = butter(2, 24, fs=fs, output='sos')
#smoothed = sosfiltfilt(sos, newset.acc)
accs = np.copy(df[:,1:7])

smoothed = np.stack([
    sosfiltfilt(sos, df[:, 1]),
    sosfiltfilt(sos, df[:, 2]),
    sosfiltfilt(sos, df[:, 3]),
    sosfiltfilt(sos, df[:, 4]),
    sosfiltfilt(sos, df[:, 5]),
    sosfiltfilt(sos, df[:, 6]),
    sosfiltfilt(sos, df[:, 7]),
    sosfiltfilt(sos, df[:, 8]),
    sosfiltfilt(sos, df[:, 9]),
], axis=1)

df[:,4:7]=df[:,4:7]*np.pi/180

time= np.array(df[:,0],dtype=mpf)#/10**9
#time = time-2*time[0]+time[1]
df[:,0]*=10**9
#time = time-2*time[0]+time[1]#
#df[:,7:10] = c_mag

#normal = np.mean(df[:100,7:10],axis=0)

from scipy.signal import savgol_filter
from matplotlib.markers import MarkerStyle
acc_smooth0 = savgol_filter(df[:,1], 500, 2)
acc_smooth1 = savgol_filter(df[:,2], 500, 2)
acc_smooth2 = savgol_filter(df[:,3], 500, 2)
acc_smooth = np.vstack((acc_smooth0,acc_smooth1,acc_smooth2)).T
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(acc_smooth)
ax.set_title('Acc smoothed')

def intersection_line_from_planes(n1, d1, n2, d2):
    """
    Trouve la droite d'intersection de deux plans
    n1, n2 : normales (vecteurs 3D)
    d1, d2 : scalaires (termes constants des équations de plan)
    Retourne : un point P sur la droite et un vecteur directeur v
    """
    n1 = np.array(n1, dtype=float)
    n2 = np.array(n2, dtype=float)
    
    # direction = produit vectoriel
    v = np.cross(n1, n2)
    
    if np.allclose(v, 0):
        raise ValueError("Les deux plans sont parallèles ou confondus (pas de droite unique).")
    
    # Résolution du système : n1·X + d1 = 0, n2·X + d2 = 0
    # On construit une matrice pour résoudre
    A = np.array([n1, n2, v])  # système avec v comme 3e équation
    b = -np.array([d1, d2, 0], dtype=float)
    
    # Résolution par moindres carrés (on force X à être solution)
    P = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return P, v

import sympy as sp
N = n_end-n_start
normals = np.zeros((N,3))
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])
gravity = [0,0,np.mean(np.linalg.norm(acc_smooth[:150,:],axis=1))]
for i in range(0,N,1):
    ss = 500
    if i<N-1-ss:
        #acc_mean = np.mean(df[i:ss+i,1:4],axis=0)
        acc_mean = np.mean(acc_smooth[i:ss+i,0:3],axis=0)
        a = np.copy(acc_mean)
        a=a/np.linalg.norm(a)
        mag_mean = np.mean(df[i:ss+i,7:10],axis=0)
        mag_mean = mag_mean /np.linalg.norm(mag_mean)
        
        acc_mean = acc_mean /np.linalg.norm(gravity)
        alpha = np.sqrt(np.max(np.abs(1-acc_mean[2]**2),0))
        
        angle_alpha= np.arccos(alpha)
        
        """angle_beta = np.pi/2-angle_alpha
        beta = np.cos(angle_beta)
        angle_gamma = np.arccos(np.sign(mag_mean[2])*np.max(np.abs(mag_mean[2]),np.abs(beta))/beta)"""
        
        qq1 = quat_ntom( np.array([1,0,0]),mag_mean)
        rotated_z = np.array(quat_rot(np.array([0,0,0,1]),(qq1)))[1:4]
        
        
        axis1 = mag_mean
        
        pacc = np.dot(rotated_z,axis1)*axis1
        oacc = rotated_z-pacc
        #normalized_oacc = oacc /np.linalg.norm(oacc)
        
        paz = rotated_z[2] - pacc[2]
        
        #beta = np.arcsin(np.linalg.norm(paz)/np.linalg.norm(rotated_z))
        #target_acc = acc_mean[2]/np.sin(beta)
        n1 =axis1
        n2 = zaxis
        d1 = -np.dot(rotated_z,axis1)
        d2 = -acc_mean[2]
        print(n1,n2,d1,d2)
        point, direction = intersection_line_from_planes(n1, d1, n2, d2)
        
        oacc= oacc/np.linalg.norm(oacc)
        direction = direction/np.linalg.norm(direction)
        t = sp.Symbol('t', real=True)
        P = sp.Matrix(point)
        d = sp.Matrix(direction)
        A = sp.Matrix(pacc)
        
        X_t = P + t * d
        R = np.linalg.norm(oacc)
        print("rotated",rotated_z,axis1)
        eq = sp.N((X_t - A).dot(X_t - A)-R**2,40)
        #eq = sp.Eq((X_t - A).dot(X_t - A), R**2)
        #solutions= sp.solve(sp.diff(eq),t)
        sol0 = sp.nsolve(sp.diff(eq),0)
        sol1 = -sol0
        
        solutions=[sol0,sol1]
        points = [np.array(X_t.subs(t, sol)).flatten() for sol in solutions]
        print(P,d,A)
        print(points)
        print("test vectors")
        print(np.dot(point,n1)+d1)
        print(np.dot(point,n2)+d2)
        tthetas= np.zeros(len(points))
        for k in range(len(points)):
            p = points[k]
            dd = p-pacc
            dd = dd/np.linalg.norm(dd.astype(float))*np.linalg.norm(oacc.astype(float))
            print("oacc,dd",oacc,dd,np.dot(oacc,dd),np.linalg.norm(np.array(dd).astype(float)))
            theta0 = np.arccos(np.sign(np.dot(oacc.astype(float),dd.astype(float)))*np.min([np.abs(np.dot(oacc.astype(float),dd.astype(float))),1]))
            theta1 = -theta0
            print(theta0,theta1)
            v0 = np.array(quat_rot([0,*oacc],ExpQua(theta0*axis1)))[1:4]
            v1 = np.array(quat_rot([0,*oacc],ExpQua(theta1*axis1)))[1:4]
            if (np.abs(np.dot(v0,np.array(dd).astype(float))))<(np.abs(np.dot(v1,np.array(dd).astype(float)))):
                print("1")
                tthetas[k] = theta1
            else:
                print("0")
                tthetas[k] = theta0
            print("rotation",np.array(quat_rot([0,*oacc],ExpQua(theta0*axis1)))[1:4],dd)
            print("rotation",np.array(quat_rot([0,*oacc],ExpQua(theta1*axis1)))[1:4],dd)
            print("thetas")
            
        
        #ge = np.cross(axis,normalized_oacc)
        print("vector",pacc,axis1,oacc,direction,np.dot(oacc,direction))
        ttheta = 0
        theta0 = tthetas[0]
        theta1 = tthetas[1]
        v0 = np.array(quat_rot([0,*oacc],ExpQua(theta0*axis1)))[1:4]
        v1 = np.array(quat_rot([0,*oacc],ExpQua(theta1*axis1)))[1:4]
        print("cmopa",v0,v1,a)
        if (np.abs(np.dot(v0,np.array(a).astype(float))))<(np.abs(np.dot(v1,np.array(a).astype(float)))):
            ttheta = theta1
        else:
            ttheta = theta0
            
        
        qq2 = quat_mult(ExpQua(ttheta*axis1), qq1)
        print("rotation",np.array(quat_rot([0,0,0,1],qq2))[1:4],a)
        normal = np.array(quat_rot([0,0,0,1], quat_inv(qq2)))[1:4]
        print("normal",normal,sol0==sol1,sol1-sol0)
    normals[i,:] = normal/np.linalg.norm(normal)


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
#proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)
Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)

#Solv2 = SolverFilterPlan(PredictFilterPlan,q0,q1,r0,r1,normal,None)   


"""for i in range(0,N-1,1):
    Solv0.update_noarg(time=time[i+1])
    #Solv0.update((i+1)/newset.freq, newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    #Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
"""
    #Solv0.update((i+1)/newset.freq, newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
newset.orient = Solv0.quaternion[:,:]  
nn=0
#neworient = newset.new_orient()
#N=2
from pyproj import Proj

from pyproj import Transformer
# Définir une projection UTM (zone 31N ici, adaptée à Paris)
proj_utm = Proj(proj="utm", zone=31, ellps="WGS84")


gps = data.values[n_start:n_end,[-3,-2]]
R = 6371000
x, y = proj_utm(gps[:,1], gps[:,0])

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32722", always_xy=True)

# Conversion (lon, lat) -> (x, y) en mètres
x, y = transformer.transform(gps[:,1], gps[:,0])

coords = np.column_stack((x, y))-np.array([x[0],y[0]])
correction_applied = np.zeros(N)
angle_applied = np.zeros(N)



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(normals)
ax.set_title('Evolution of the normal')
     
sos = butter(2, 2, fs=fs, output='sos')
smoothed = np.stack([
    sosfiltfilt(sos, normals[:, 0]),
    sosfiltfilt(sos, normals[:, 1]),
    sosfiltfilt(sos, normals[:, 2]),
], axis=1)
normals = smoothed


y = normals

# Savitzky-Golay (fenêtre=11, polynôme=3)
y_smooth0 = savgol_filter(y[:,0], 500, 2)
y_smooth1 = savgol_filter(y[:,1], 500, 2)
y_smooth2 = savgol_filter(y[:,2], 500, 2)
y_smooth = np.vstack((y_smooth0,y_smooth1,y_smooth2)).T



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(y_smooth)
ax.set_title('Evolution of the normal')
normals = np.copy(y_smooth) 
for i in range(0,N-1,1):
    
    nn+=1
    normal = normals[i+1,:]
    print("iteration",i)
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
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
ax.legend(['Z coordinate of gravity vector returned by Rev-MEKF','Z coordinate of Accelerometer',])
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
        row[f"acc_x_{k}"] = newset.acc[j,0]
    for k, j in enumerate(indices):
        row[f"acc_y_{k}"] = newset.acc[j,1]
    for k, j in enumerate(indices):
        row[f"acc_z_{k}"] = newset.acc[j,2]

    # Gyro
    for k, j in enumerate(indices):
        row[f"gyro_x_{k}"] = newset.gyro[j,0]
    for k, j in enumerate(indices):
        row[f"gyro_y_{k}"] = newset.gyro[j,1]
    for k, j in enumerate(indices):
        row[f"gyro_z_{k}"] = newset.gyro[j,2]

    # Mag
    for k, j in enumerate(indices):
        row[f"mag_x_{k}"] = newset.mag[j,0]
    for k, j in enumerate(indices):
        row[f"mag_y_{k}"] = newset.mag[j,1]
    for k, j in enumerate(indices):
        row[f"mag_z_{k}"] = newset.mag[j,2]
        
        
    for k, j in enumerate(indices):
        row[f"normal_x_{k}"] = normals[j,0]
    for k, j in enumerate(indices):
        row[f"normal_y_{k}"] = normals[j,1]
    for k, j in enumerate(indices):
        row[f"normal_z_{k}"] = normals[j,2]


    rows.append(row)
df = pd.DataFrame(rows)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"corrections_windows_{timestamp}"+ str(n_start)+".csv", index=False)