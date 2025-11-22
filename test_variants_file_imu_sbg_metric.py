
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
#data_file = 'imu_data_sbg_static_120925.csv'
#moving
#data_file= 'imu_data_sbg_250701_1344.csv'
#data_file= 'imu_data_sbg_250701_1628.csv'
#data_file = 'imu_data_sbg_plane_09121118.csv'
#data_file = 'imu_data_sbg_plane_09121136.csv'
#data_file = 'imu_data_sbg_plane_09121245.csv'
#data_file = 'imu_data_fall_09131902.csv'
#data_file = 'imu_data_vertical_inhand_09131913.csv'
#data_file = 'imu_data_sbg_car_09141253.csv'
#data_file = 'imu_data_sbg_plane_09121311.csv'
#slope
#data_file= 'imu_data_sbg_250701_1710.csv'

#data_file = 'odo_data.csv'

#dataset
#data_file = 'dataset_gps_mpu_left.csv'


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
n_end=n_start +600
cols = np.array([0,1,2,3,10,11,12,19,20,21])
df = data.values[n_start:n_end,:]


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
df[:,4:7]=df[:,4:7]#*np.pi/180

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
time= np.array(df[:,0],dtype=mpf)/10**9
#time = time-2*time[0]+time[1]
#df[:,0]*=10**9
#time = time-2*time[0]+time[1]#
#df[:,7:10] = c_mag

normal = np.mean(df[:100,7:10],axis=0)

newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23)#,normal=np.array([0.1101,1,0])) 
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
q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(-1), 10**(-1)
#normal = np.array([-1,1,0],dtype=mpf)10
normal = newset.normal

#normal[2] = 0  
#normal = normal/np.linalg.norm(normal)

proj_func = correct_proj2
proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)#,grav=newset.grav)
#q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(0), 10**(-2)

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
correction_applied = np.zeros(N)

for i in range(0,N-1,1):
    nn+=1
    print("iteration",i)
    #Solv0.update_noarg(time=time[i+1])
    #Solv1.update_noarg(time=time[i+1])
    #Solv2.update_noarg(time=time[i+1])
    Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    correction_applied[i] = Solv2.KFilter.corrected

    #print(Solv0.KFilter.speed,Solv1.KFilter.speed)
    
    if i%10 ==0 and i>0:
        """fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(Solv1.position[i-10:i,:])
        ax.set_title('position1')"""
    
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        #ax.plot([np.linalg.norm(Solv0.position[i-10+j,:]) for j in range(10)])
        ax.plot([np.linalg.norm(Solv1.position[i-10+j,:]) for j in range(10)])
        ax.plot([np.linalg.norm(Solv2.position[i-10+j,:]) for j in range(10)])
        
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
for i in range(N):
    ground_truth[i,:] = q_mean

for i in range(1,N):
    metric0[i] = metric0[i-1] + np.abs(1-quat_mult(quaternion0[i,:],quat_inv(ground_truth[i,:]))[0])
    metric1[i] = metric1[i-1] + np.abs(1-quat_mult(quaternion1[i,:],quat_inv(ground_truth[i,:]))[0])
    metric2[i] = metric2[i-1] + np.abs(1-quat_mult(quaternion2[i,:],quat_inv(ground_truth[i,:]))[0])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],metric1-metric2)
ax.plot(time0[np.argwhere(correction_applied).flatten()], [np.linalg.norm((metric1-metric2)[j]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

#ax.plot(metric2)
ax.legend(['Lambda(X_MEKF,T) - Lambda(X_REVMEKF,T)','Correction applied'])
#plt.yscale("log")
plt.xlabel('Seconds')
plt.ylabel('Cumulated error')
ax.set_title('Difference of the metric computed by MEKF and Heuristical Rev-MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],metric1)
ax.plot(time0[:size],metric2)
ax.legend(['Lambda(X_MEKF,T)','Lambda(X_REVMEKF,T)'])
plt.yscale("log")
plt.xlabel('Seconds')
plt.ylabel('Cumulated error')
ax.set_title('Metric computed by MEKF and Heuristical Rev-MEKF')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],metric1)
ax.plot(time0[:size],metric2)
ax.legend([r'$\Lambda(X_\text{MEKF},\mathcal{T})$',r'$\Lambda(X_\text{Rev-MEKF},\mathcal{T})$'],fontsize=14)
plt.yscale("log")
plt.xlabel('Seconds')
plt.ylabel('Cumulated error')
ax.set_title('Metric computed by MEKF and Heuristical Rev-MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size],metric1-metric2)
ax.plot(time0[np.argwhere(correction_applied).flatten()], [((metric1-metric2)[j]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

#ax.plot(metric2)
ax.legend([r'$\Lambda(X_\text{MEKF},\mathcal{T}) - \Lambda(X_\text{Rev-MEKF},\mathcal{T})$','Correction applied'],fontsize=14)
#plt.yscale("log")
plt.xlabel('Seconds')
plt.ylabel('Cumulated error')
ax.set_title('Difference of the metric computed by MEKF and Heuristical Rev-MEKF')

acc_ext = acc_earth-Solv2.KFilter.gravity
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],np.linalg.norm(acc_ext,axis=1))
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