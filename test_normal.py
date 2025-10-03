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


#data_file= 'imu_data_sbg_250630_2340_acconly.csv'
#data_file = 'imu_data_rosbag.csv'
#data_file = 'planar_movement_250625_1057.csv'
#data_file = 'static_250625_2.csv'
#data_file = 'static_260625.csv'

#file_name = data_file.split('imu')[1]
#mag_file = 'mag'+file_name

#N = 10000
data=pd.read_csv(data_file)


'''leica_file = 'leica_data.csv'

leica=pd.read_csv(leica_file)
leica_values = leica.values
time_leica = leica_values[:,0]/10**9'''

#mmode = 'OdoAccPre'
mmode = 'GyroAccMag'
#mmode = 'GyroAccMag'
#mmode = 'GyroAcc'

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

#mag=pd.read_csv(mag_file, header=None)
#print(data.head())
n_start = 0
n_end=4000
n_end=n_start +10000
cols = np.array([0,1,2,3,10,11,12,19,20,21])
df = data.values[n_start:n_end,cols]

acc = np.copy(df[:,1:4])
mag = np.copy(df[:,7:10])

"""df[:,1]=acc[:,1]
df[:,2]=acc[:,0]
df[:,3]=-acc[:,2]

df[:,7]=mag[:,1]
df[:,8]=mag[:,0]
df[:,9]=-mag[:,2]
"""
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
#df[:,1:10] = smoothed
#df[:,7:10]=smoothed[:,6:9]
df[:,4:7]=df[:,4:7]*np.pi/180

'''smoothed = np.stack([
    sosfiltfilt(sos, df[:, 1]),
    sosfiltfilt(sos, df[:, 2]),
    sosfiltfilt(sos, df[:, 3]),
    sosfiltfilt(sos, df[:, 4]),
    sosfiltfilt(sos, df[:, 5])
], axis=1)
df[:,1:6] = smoothed'''

'''accx = df[:,4]
accy = -df[:,3]
accz = df[:,5]
df[:,3:6] = np.vstack([accx,accy,accz]).T'''

"""fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(smoothed)
ax.set_title("acc")
"""
#gyro = np.array(data.values[:,4:7],dtype=mpf)
#g_bias = np.mean(gyro[:50,:],axis=0)
#c_mag = mag.values[n_start:n_end,:]

#df = data.values[n_start:n_end,:]
#startleic = leica_values[n_start:n_end,3]
#df[:,6] = startleic
#df[:,0]=df[:,0]-df[0,0]
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
normals = np.zeros((N,3))
for i in range(0,N,1):
    ss = 500
    if i<N-1-ss:
        #acc_mean = np.mean(df[i:ss+i,1:4],axis=0)
        acc_mean = np.mean(acc_smooth[i:ss+i,0:3],axis=0)
        acc_mean[2] = acc_mean[2] /np.linalg.norm(gravity)
        alpha = np.sqrt(np.max(np.abs(1-acc_mean[2]**2),0))
        #acc_mean = acc_mean/np.linalg.norm(acc_mean)
        #print("alpha",alpha,1-acc_mean[2]**2)
        acc_mean[0:2] = acc_mean[0:2]/np.linalg.norm(acc_mean[0:2])*alpha
        #acc_mean = acc_mean/np.linalg.norm(acc_mean)
        acc_angle = np.arctan2(acc_mean[0],acc_mean[1])#+np.arctan2(1,0)
        
        acc_reoriented = np.array(quat_rot([0,*acc_mean],ExpQua(zaxis*acc_angle)))[1:4]
        normal0 = np.array(quat_rot([0,0,0,1],(quat_ntom(acc_reoriented,np.array([0,0,1])))))[1:4]
        
        qq_final0 = quat_mult((quat_ntom(np.array([0,0,1]),normal0)),(ExpQua(zaxis*acc_angle)))
        #print(normal0,np.linalg.norm(acc_mean))
        
        mag_mean = np.mean(df[i:ss+i,7:10],axis=0)
        mag_mean = mag_mean /np.linalg.norm(mag_mean)
        
        a = acc_mean
        a = a/mp.norm(a)
        m = mag_mean
        m = m/mp.norm(m)
        adm = skewSymmetric(a)@m
        adm = adm/mp.norm(adm)
        new_m = skewSymmetric(adm)@a
        M = np.array([-skewSymmetric(a)@m,m,a]).T
        #M = np.array([m,skewSymmetric(a)@m,a]).T
        #M = np.array([new_m,adm,a]).T
        quat = normalize(quat_inv(RotToQuat(M)))
        #print("result",np.array(quat_rot([0,0,0,1], quat))[1:4])
        normal = np.array(quat_rot([0,0,0,1], quat))[1:4]
    normals[i,:] = normal/np.linalg.norm(normal)


newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23,normals=normals)#,gravity=np.array([0,0,9.80665],dtype=mpf))#,normal=np.array([0.1101,1,0])) 
N=newset.size
#N=len(df)
nn = N-1
g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)
angle = int(N/2)
#newset = KFilterData(N,mpf(1.),alpha=angle) 
"""normal_vertical = np.array([mpf('0.004255930241527828857604722545936216541889906'), mpf('-0.003213834762718947045532726200938664393319883'), mpf('0.9999857790608310386391832597563166216076963')],dtype=mpf)
normal_slope = np.array([mpf('-0.2048232814136505330643787156746757575889839'), mpf('0.1392655006235043046926350641747759506549044'), mpf('0.9688408247627828168544761849522810335526499')],dtype=mpf)
qq_vertical = np.array([mpf('0.0005238042742003194250313644461650544515406238'), mpf('0.952500508020517862402287684560623411714904'), mpf('-0.3045310020704403439306087292097119761367074'), mpf('0.001837560260556293121729518756430819274065758')],dtype=mpf)
qq_slope = np.array([mpf('0.03530534031735835920108235692000486469505786'), mpf('0.9572305149107438470437330713031977439517386'), mpf('-0.2607293604697206371253035678210890253121073'), mpf('0.1203473093086762345391241175000634127777685')],dtype=mpf)
q_var = quat_mult(qq_slope,quat_inv(qq_vertical))
qq = quat_ntom(normal_vertical, np.array([0,0,1]))
slope = np.array(quat_rot([0,0,0,1],q_var))[1:4]"""

#best_cand = np.array([ 5.10742705e-03,  7.13755855e-05, -9.98163928e-01],dtype=mpf)

#newset = KFilterDataFile(df,mode=mmode,surf=np.array([1,0,0]),normal=np.array([0,0,-1]))#,normal=slope)#,normal=np.array([0,0,1]))
#newset = KFilterData(N,mpf(10)/mpf(1),g_bias= 10**(-2),g_noise=10**(-10),a_noise=10**(-4)) 
orient = newset.orient
pos_earth = newset.pos_earth

q0,q1,r0,r1 = 10**(-5), 10**(-5), 10**(8), 10**(-2) 
q0,q1,r0,r1 = 10**(-4), 10**(-4), 10**(-2), 10**(-1)
q0,q1,r0,r1 = 10**(-4), 10**(-4), 10**(-2), 10**(-1)
q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(0), 10**(0)
#normal = np.array([-1,1,0],dtype=mpf)10
normal = newset.normal

xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])
gravity = newset.gravity
for i in range(10):
    acc_mean = np.mean(df[i:100+i,1:4],axis=0)
    acc_mean[2] = acc_mean[2] /np.linalg.norm(gravity)
    alpha = np.sqrt(1-acc_mean[2]**2)
    #acc_mean = acc_mean/np.linalg.norm(acc_mean)
    
    acc_mean[0:2] = acc_mean[0:2]/np.linalg.norm(acc_mean[0:2])*alpha
    #acc_mean = acc_mean/np.linalg.norm(acc_mean)
    acc_angle = np.arctan2(acc_mean[0],acc_mean[1])#+np.arctan2(1,0)
    
    acc_reoriented = np.array(quat_rot([0,*acc_mean],ExpQua(zaxis*acc_angle)))[1:4]
    normal0 = np.array(quat_rot([0,0,0,1],(quat_ntom(acc_reoriented,np.array([0,0,1])))))[1:4]
    
    qq_final0 = quat_mult((quat_ntom(np.array([0,0,1]),normal0)),(ExpQua(zaxis*acc_angle)))
    #print(normal0,np.linalg.norm(acc_mean))
    
    mag_mean = np.mean(df[i:100+i,7:10],axis=0)
    mag_mean = mag_mean /np.linalg.norm(mag_mean)
    
    a = acc_mean
    a = a/mp.norm(a)
    m = mag_mean
    m = m/mp.norm(m)
    M = np.array([-skewSymmetric(a)@m,m,a]).T
    quat = normalize(quat_inv(RotToQuat(M)))
    #print(np.array(quat_rot([0,0,0,1], quat))[1:4])
    """
    mag_angle = np.arctan2(mag_mean[0],mag_mean[1])#+np.arctan2(1,0)
    
    mag_reoriented = np.array(quat_rot([0,*mag_mean],ExpQua(zaxis*mag_angle)))[1:4]
    normal1_0 = np.array(quat_rot([0,0,0,1],(quat_ntom(mag_reoriented,np.array([1,0,0])))))[1:4]
    normal1 = np.array([0,0,normal1_0[2]])
    normal1[0] = np.sqrt(1-normal1[2]**2)
    
    qq_final1 = quat_mult((quat_ntom(np.array([0,0,1]),normal1)),(ExpQua(zaxis*mag_angle)))
    print(normal0,normal1,normal1_0)
    print(mag_reoriented,mag_mean)
    print(acc_angle,mag_angle)"""
    
    


proj_func = correct_proj2
#proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)
#q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(-1), 10**(-1)
#q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(0), 10**(-1)

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
    normal = newset.normal
    print("iteration",i)
    #Solv0.update_noarg(time=time[i+1])
    #Solv1.update_noarg(time=time[i+1])
    #Solv2.update_noarg(time=time[i+1])
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)

    Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
    #print(Solv0.KFilter.speed,Solv1.KFilter.speed)
    correction_applied[i] = Solv2.KFilter.corrected
    angle_applied[i+1] =angle_applied[i]+Solv2.KFilter.angle
    
    

    if i%10 ==0 and i>0:
        """fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(Solv1.position[i-10:i,:])
        ax.set_title('position1')"""
    
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
        "correction_applied": p0 in np.argwhere(correction_applied).flatten()
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


    rows.append(row)
df = pd.DataFrame(rows)

df.to_csv("corrections_windows_"+ str(n_start)+".csv", index=False)