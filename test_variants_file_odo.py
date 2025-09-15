
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

from proj_func import correct_proj2,correct_proj3
from function_quat import *
from kfilterdata_file_odo import KFilterDataFile


mp.dps = 40

import sys,os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/examples')[0])


from scipy.spatial.transform import Rotation
from solver_kalman import SolverFilterPlan


data_file = 'imu_data5.csv'
data_file = 'calibration_data_250523.csv'
data_file = 'calibration_020625.csv'
data_file = 'calibration_030625.csv'
data_file = 'imu_data_sbg_250624_2055.csv'


data_file = 'imu_data_sbg_250624_2259.csv'
data_file = 'imu_data_sbg_250624_2055.csv'

#static
data_file= 'imu_data_sbg_250701_1120.csv'
#moving
#data_file= 'imu_data_sbg_250701_1344.csv'
#data_file= 'imu_data_sbg_250701_1628.csv'
#slope
data_file= 'imu_data_sbg_250701_1710.csv'

data_file = 'odo_data_07251434.csv'
data_file = 'odo_data_07251436.csv'
data_file = 'odo_data_07251437.csv'
#data_file = 'odo_data.csv'
#data_file= 'imu_data_sbg_250630_2340_acconly.csv'
#data_file = 'imu_data_rosbag.csv'
#data_file = 'planar_movement_250625_1057.csv'
#data_file = 'static_250625_2.csv'
#data_file = 'static_260625.csv'

#file_name = data_file.split('imu')[1]
#mag_file = 'mag'+file_name

#N = 10000
data=pd.read_csv(data_file)

leica_file = 'leica_data_07251434.csv'
leica_file = 'leica_data_07251436.csv'
leica_file = 'leica_data_07251437.csv'
#leica_file = 'leica_data.csv'
leica=pd.read_csv(leica_file)
leica_values = leica.values
time_leica = leica_values[:,0]/10**9

mmode = 'OdoAccPre'
#mmode = 'GyroAccMag'
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
n_start = 100
n_end=4000
n_end=n_start+500
df = data.values[n_start:n_end,:]


fs = 50
sos = butter(2, 20, fs=fs, output='sos')
#smoothed = sosfiltfilt(sos, newset.acc)
"""smoothed = np.stack([
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
df[:,1:10] = smoothed"""

smoothed = np.stack([
    sosfiltfilt(sos, df[:, 1]),
    sosfiltfilt(sos, df[:, 2]),
    sosfiltfilt(sos, df[:, 3]),
    sosfiltfilt(sos, df[:, 4]),
    sosfiltfilt(sos, df[:, 5])
], axis=1)
df[:,1:6] = smoothed
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[:,3:6])
g = 9.80
accx = df[:,4]*g
accy = -df[:,3]*g
accz = df[:,5] *g
df[:,3:6] = np.vstack([accx,accy,accz]).T

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[:,3:6])

quats = np.zeros((len(df[:,3:6]),4))
for i in range(len(df[:,3:6])):
    accx,accy,accz = df[i,3:6]
    p_acc = np.array([accx,accy,0])
    acc = np.array([accx,accy,accz])
    qq = quat_ntom(acc/np.linalg.norm(acc), p_acc/np.linalg.norm(p_acc))
    quats[i,:] = quat_inv(qq)
    #df[i,3:6] = np.array(quat_rot([0,*acc],qq))[1:4]
    
    
"""fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[:,3:6])"""
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quats)
q_calib = np.mean(quats[:100,:],axis=0)
q_calib = q_calib/np.linalg.norm(q_calib)
normal0 = np.array(quat_rot([0,-1,0,0],(q_calib)))[1:4]
normal0[2] = normal0[2]**2+normal0[1]**2
normal0[1] = 0

#df[:,1:3] = np.fliplr(df[:,1:3])
"""fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(smoothed)
ax.set_title("acc")
"""
#gyro = np.array(data.values[:,4:7],dtype=mpf)
#g_bias = np.mean(gyro[:50,:],axis=0)
#c_mag = mag.values[n_start:n_end,:]

#df = data.values[n_start:n_end,:]
startleic = leica_values[n_start:n_end,3]
df[:,6] = startleic
time= np.array(df[:,0],dtype=mpf)/10**9
#df[:,7:10] = c_mag
#newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23,surf=np.array([-1,0,0.35])) 
#newset = KFilterDataFile(df[:,:],mode=mmode,base_width=0.23,surf=np.array([-1,0,0])) 
newset = KFilterDataFile(df[:,:],mode=mmode,base_width=0.23,surf=np.array(normal0))
#newset = KFilterDataFile(df[:,:],mode=mmode,base_width=0.23,surf=np.array([-1,0,0])) 
N=newset.size
#N=len(df)
nn = N-1
g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)
angle = int(N/2)

orient = newset.orient
pos_earth = newset.pos_earth

q0,q1,r0,r1 = 10**(-5), 10**(-5), 10**(8), 10**(-2) 
q0,q1,r0,r1 = 10**(-4), 10**(-4), 10**(-2), 10**(-1)
q0,q1,r0,r1 = 10**(-1), 10**(-1), 10**(-2), 10**(-2)
q0,q1,r0,r1 = 10**(-1), 10**(-1), 10**(-1), 10**(-1)
q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(-2), 10**(-2)
#normal = np.array([-1,1,0],dtype=mpf)10
normal = newset.normal



proj_func = correct_proj3
#proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=None)
Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)#,grav=newset.grav)
Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)#,grav=newset.grav)

#Solv2 = SolverFilterPlan(PredictFilterPlan,q0,q1,r0,r1,normal,None)   


#for i in range(0,N-1,1):
#    Solv0.update_noarg(time=time[i+1])
    #Solv0.update((i+1)/newset.freq, newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    #Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)

    #Solv0.update((i+1)/newset.freq, newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)

nn=0
#neworient = newset.new_orient()
#N=2
for i in range(0,N-1,1):
    nn+=1
    print("iteration",i)
    Solv0.update_noarg(time=time[i+1])
    Solv1.update_noarg(time=time[i+1])
    Solv2.update_noarg(time=time[i+1])
    #Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    #Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    #Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], newset.normal)
    
    if i%50 ==0 and i>0:
        """fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(Solv1.position[i-10:i,:])
        ax.set_title('position1')"""
    
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot([(Solv1.position[i-50+j,:]) for j in range(50)])
        
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot([(Solv2.position[i-50+j,:]) for j in range(50)])
        
        plt.show()


newset.orient = Solv0.quaternion[:,:]   
compare = np.zeros((N,4),dtype=float)
compare2 = np.zeros((N,4),dtype=float)
quaternion0 = Solv0.quaternion[:N,:]    
quaternion1 = Solv1.quaternion[:N,:]
quaternion2 = Solv2.quaternion[:N,:]
position0 = Solv0.position[:N,:]
position1 = Solv1.position[:N,:]
position2 = Solv2.position[:N,:]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion0[:,0],quaternion0[:,1],'*')
ax.plot(quaternion2[:,0],quaternion2[:,1],'*')

ax.legend(['Input quaternion','Resulting quaternion'])
ax.set_title('Comparison of initial quaternion and result with freq=' + str(newset.freq))


for i in range(N):
    #print(quat_mult(quaternion[i,:],quat_inv(orient[i,:])))
    compare[i,:] = normalize(quat_mult(quaternion0[i,:],quat_inv(orient[i,:])))-np.array([1,0,0,0])
    compare2[i,:] = normalize(quat_mult(quaternion2[i,:],quat_inv(orient[i,:])))-np.array([1,0,0,0])
    #compare2[i,:] = quaternion2[i,:]-orient[i,:]

nn=int(N/2)    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(orient[:nn,0],orient[:nn,3],'*')
ax.plot(quaternion0[:nn,0],quaternion0[:nn,3],'*')
ax.plot(quaternion2[:nn,0],quaternion2[:nn,3],'*')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(orient[:nn,0],orient[:nn,1],'*')
ax.plot(quaternion0[:nn,0],quaternion0[:nn,1],'*')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(orient[nn:,0],orient[nn:,3],'*')
ax.plot(quaternion0[nn:,0],quaternion0[nn:,3],'*')
ax.plot(quaternion2[:nn,0],quaternion2[:nn,3],'*')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(orient[nn:,0],orient[nn:,1],'*')
ax.plot(quaternion1[nn:,0],quaternion1[nn:,1],'*')

ax.legend(['Input quaternion','Resulting quaternion'])
ax.set_title('Comparison of initial quaternion and result with freq=' + str(newset.freq))

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([mp.norm(compare[i,:]) for i in range(len(compare))],'*')
ax.plot([mp.norm(compare2[i,:]) for i in range(len(compare2))],'*')
ax.set_title('angle = ' + str(angle) + ' rad, bias =' +str(g_bias) + ', gyro noise =' + str(g_noise) + ', acc noise =' + str(a_noise))


#print(list(map(surf,position2)))

diff_pos1 = [np.linalg.norm(a) for a in position1-pos_earth]
diff_pos2 = [np.linalg.norm(a) for a in position2-pos_earth]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(diff_pos1)

ax.set_title('Diff Position with EKF with Reality in 3D, ProjAlgo')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(diff_pos2)

ax.set_title('Diff Position with EKF with Reality in 3D, CorrProjAlgo')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2-pos_earth)

ax.set_title('Diff Position with EKF with Reality in 3D, CorrProjAlgo')


"""fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([mp.norm(compare[i,:]) for i in range(len(compare))],'*')
ax.plot([mp.norm(compare2[i,:]) for i in range(len(orient))],'*')
ax.set_title('angle = ' + str(angle) + ' rad, bias =' +str(g_bias) + ', gyro noise =' + str(g_noise) + ', acc noise =' + str(a_noise))"""

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([mp.norm(compare[i,:]) for i in range(len(compare[:,:]))],'*')
ax.plot([mp.norm(compare2[i,:]) for i in range(len(orient))],'*')
ax.legend(['ProjAlgo','CorrProjAlgo'])

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot(compare2)

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
ax.plot(newset.orient)
ax.set_title('orient')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0)
ax.set_title('position0')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position1)
ax.set_title('position1')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2)
ax.set_title('position2')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth)
ax.set_title('pos_earth')



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position0)
ax.set_title('dpos_earth0')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position1)
ax.set_title('dpos_earth1')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position2)
ax.set_title('dpos_earth2')

compare0 = np.zeros((N,4))
for ind in range(N):
    quat=newset.orient[ind,:]
    compare0[ind,:] = normalize(quat_mult(quaternion0[ind,:],quat_inv(newset.orient[ind,:])))-np.array([1,0,0,0])


compare1 = np.zeros((N,4))
quat = newset.rotsurf
for ind in range(N):
    #quat=newset.orient[ind,:]
    compare1[ind,:] = normalize(quat_mult(quaternion1[ind,:],quat_inv(newset.orient[ind,:])))-np.array([1,0,0,0])
    #compare1[ind,:] = normalize(quat_mult(quaternion1[ind,:],quat_inv(quat)),quat_inv(newset.orient[ind,:])))-np.array([1,0,0,0])


compare2 = np.zeros((N,4))
quat = newset.rotsurf
for ind in range(N):
    #quat=newset.orient[ind,:]
    compare2[ind,:] = normalize(quat_mult(quaternion2[ind,:],quat_inv(newset.orient[ind,:])))-np.array([1,0,0,0])
    #compare1[ind,:] = normalize(quat_mult(quaternion1[ind,:],quat_inv(quat)),quat_inv(newset.orient[ind,:])))-np.array([1,0,0,0])
compare3 = np.zeros((N,4))
quat = newset.rotsurf
for ind in range(N):
    #quat=newset.orient[ind,:]
    compare3[ind,:] = normalize(quat_mult(quaternion2[ind,:],quat_inv(quaternion1[ind,:])))-np.array([1,0,0,0])
    #compare1[ind,:] = normalize(quat_mult(quaternion1[ind,:],quat_inv(quat)),quat_inv(newset.orient[ind,:])))-np.array([1,0,0

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare0)
ax.set_title('compare0')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare1)
ax.set_title('compare1')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare2)
ax.set_title('compare2')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare3)
ax.set_title('compare3')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth[:,1],pos_earth[:,2])
ax.set_title('True trajectory')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0[:,1],position0[:,2])
ax.set_title('Position with integration of gyroscope')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position1[:,1],position1[:,2])
ax.set_title('Position with MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2[:,1],position2[:,2])
ax.set_title('Position with Reversible MEKF')




#leica_file = 'leica_data_07251437.csv'

#N = 10000
leica=pd.read_csv(leica_file)
df = leica.values
rel_df = df[n_start:n_end,1:]-df[n_start,1:]

for i in range(len(rel_df)):
    lei = rel_df[i,:]
    p_lei = np.array([0,-lei[1],lei[2]])
    if np.linalg.norm(p_lei) > 0:
        qq = quat_ntom(lei/np.linalg.norm(lei), p_lei/np.linalg.norm(p_lei))
        rel_df[i,:] = np.array(quat_rot([0,*lei],qq))[1:4]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(rel_df)

df[n_start:n_end,1:] = rel_df

time_leica = df[:,0]/10**9
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:])
#ax.plot(time[1:]-time[0],position1[1:,0])
ax.set_title('leica pos')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,3])
ax.plot(time[1:]-time[0],position0[1:,2])
ax.plot(time[1:]-time[0],position1[1:,2])
ax.plot(time[1:]-time[0],position2[1:,2])
ax.set_title('leica vs odo')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0],df[n_start:n_end,3])
ax.plot(time[1:]-time[0],position0[1:,2])
ax.plot(time[1:]-time[0],position1[1:,2])
ax.plot(time[1:]-time[0],position2[1:,2])
ax.set_title('leica vs odo')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.pressure)



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time[1:]-time[0],position2[1:,:])

ax.set_title('odo2')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:])

ax.set_title('leica')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1])
ax.plot(time[1:]-time[0],position0[1:,0])
ax.plot(time[1:]-time[0],position1[1:,0])
ax.plot(time[1:]-time[0],position2[1:,0])   

ax.set_title('leicax vs odox')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1])
ax.plot(time[1:]-time[0],position0[1:,0])
ax.plot(time[1:]-time[0],position1[1:,0])
ax.plot(time[1:]-time[0],position2[1:,0])

ax.set_title('leicax vs odox')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)
ax.set_title('acc')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:]-df[n_start,1:])
ax.plot(time[1:]-time[0],position2[1:,:])

ax.set_title('leica vs odo')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:]-df[n_start,1:])
ax.plot(time[1:]-time[0],position1[1:,:])

ax.set_title('leica vs odo')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:]-df[n_start,1:])
ax.plot(time[1:]-time[0],position2[1:,:])

ax.set_title('leica vs odo2')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,3]-df[n_start,3])
ax.plot(time[1:]-time[0],position1[1:,2])

ax.set_title('leica vs odo1 z')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,2]-df[n_start,2])
ax.plot(time[1:]-time[0],position1[1:,1])

ax.set_title('leica vs odo1 y')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1]-df[n_start,1])
ax.plot(time[1:]-time[0],position1[1:,0])

ax.set_title('leica vs odo1 x')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,3]-df[n_start,3])
ax.plot(time[1:]-time[0],position2[1:,2])

ax.set_title('leica vs odo2 z')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,2]-df[n_start,2])
ax.plot(time[1:]-time[0],position2[1:,1])

ax.set_title('leica vs odo2 y')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1]-df[n_start,1])
ax.plot(time[1:]-time[0],position2[1:,0])

ax.set_title('leica vs odo2 x')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,3]-df[n_start,3])
ax.plot(time[1:]-time[0],position0[1:,2])

ax.set_title('leica vs odo0 z')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,2]-df[n_start,2])
ax.plot(time[1:]-time[0],position0[1:,1])

ax.set_title('leica vs odo0 y')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1]-df[n_start,1])
ax.plot(time[1:]-time[0],position0[1:,0])

ax.set_title('leica vs odo0 x')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[500:,1:]-df[500,1:])

ax.set_title('leica')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,2]-df[n_start,2])
ax.plot(position2[1:,1])

ax.set_title('leica vs odo2 y')



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,3]-df[n_start,3])
ax.plot(position0[1:,2])

ax.set_title('leica vs odo0 z')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,3]-df[n_start,3])
ax.plot(position1[1:,2])

ax.set_title('leica vs odo1 z')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,3]-df[n_start,3])
ax.plot(position2[1:,2])

ax.set_title('leica vs odo2 z')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,2]-df[n_start,2])
ax.plot(position0[1:,1])

ax.set_title('leica vs odo0 y')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,2]-df[n_start,2])
ax.plot(position1[1:,1])

ax.set_title('leica vs odo1 y')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,2]-df[n_start,2])
ax.plot(position2[1:,1])

ax.set_title('leica vs odo2 y')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,1]-df[n_start,1])
ax.plot(position0[1:,0])

ax.set_title('leica vs odo0 x')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,1]-df[n_start,1])
ax.plot(position1[1:,0])

ax.set_title('leica vs odo1 x')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[n_start:n_end,1]-df[n_start,1])
ax.plot(position2[1:,0])

ax.set_title('leica vs odo2 x')


all_points = df[n_start:n_end,1:]-df[n_start,1:]
all_points  = df[n_start:n_end,1:]
d_all_points = np.diff(all_points,axis=0)

cross = np.zeros((len(d_all_points)-1,3))
for i in range(len(cross)):
    cross[i,:] = np.cross(d_all_points[i],d_all_points[i+1])/np.linalg.norm(d_all_points[i])/d_all_points[i+1]
    cross[i,:] = cross[i,:]/np.linalg.norm(cross[i,:])
    
    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc[:,0]/newset.acc[:,2])


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(np.abs(df[n_start:n_end,2]-df[n_start,2]-position1[:,1]))
ax.plot(np.abs(df[n_start:n_end,2]-df[n_start,2]-position2[:,1]))

ax.set_title('leica vs odo y')




position3= np.copy(position2)

for i in range(len(position2)):
    lei = position2[i,:]
    p_lei = np.array([0,lei[1],lei[2]])
    if np.linalg.norm(p_lei) > 0:
        qq = quat_ntom(lei/np.linalg.norm(lei), p_lei/np.linalg.norm(p_lei))
        position3[i,:] = np.array(quat_rot([0,*lei],qq))[1:4]


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,position0)
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:])
ax.set_title('Position of Integration of Odometry')
        
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,position1)
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:])
ax.set_title('Position of MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,position2)
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:])
ax.set_title('Position of Rev-MEKF')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:])
#ax.plot(time[1:]-time[0],position1[1:,0])
ax.set_title('Position from Leica')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,[np.linalg.norm(p) for p in df[n_start:n_end,1:]-position1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,[np.linalg.norm(p) for p in df[n_start:n_end,1:]-position3])
#ax.plot(time[1:]-time[0],position1[1:,0])
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:]-position1)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time_leica[n_start:n_end]-time_leica[0]-20,df[n_start:n_end,1:]-position2)
