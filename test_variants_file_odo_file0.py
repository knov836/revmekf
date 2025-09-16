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


data_file = 'odo_data_07251434_2.csv'
data_file = 'odo_data_07251437_2.csv'
data=pd.read_csv(data_file)
leica_file = 'leica_data_07251434.csv'
leica_file = 'leica_data_07251437.csv'
leica=pd.read_csv(leica_file)
leica_values0 = leica.values

leica_values = leica_values0[:,:]-leica_values0[0,:]

time_leica = leica_values0[:,0]/10**9

mmode = 'OdoAccPre'

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
"""n_start = 100
n_end=4000
n_end=n_start+500"""



#data=pd.read_csv(data_file)
N = len(data.values)
dx = data.values[:,1]
dy = data.values[:,2]
ndx = data.values[:,1]
ndy = data.values[:,2]
thetasw = data.values[:,3]
thetasz = data.values[:,4]

ddx = np.zeros(N)
ddy = np.zeros(N)
leftw = np.zeros(N)
rightw = np.zeros(N)
x0 = dx[0]
y0 = dy[0]
x = x0
y = y0
time_odo = np.array([mp.mpmathify(data.values[i,0]) for i in range(len(data.values[:,0]))])
time = time_odo.astype(float)
dtime_odo = np.diff(time_odo)

thetas = np.zeros(N)
nthetas = np.zeros(N)
dds = np.zeros(N)
for i in range(N):
    thetas[i] = np.arctan2(thetasz[i],thetasw[i])*2
nthetas = np.copy(thetas)
dthetas = np.zeros(N)
#dthetas = np.diff(thetas)
    
    
for i in range(N-1):
    d = np.linalg.norm(np.array([ddx[i-1],ddy[i-1],0]))
    theta = thetas[i]
    dtheta = dthetas[i-1]
    dds[i] = np.linalg.norm([ddx[i],ddy[i]])*np.sign(np.dot(np.array([d*mp.cos(theta+dtheta/2),d*mp.sin(theta+dtheta/2),mpf(0)]),np.array([ddx[i-1],ddy[i-1],0])))

orientation = np.zeros(6)
orientation[:3] = np.array([dx[0],dy[0],0])
orientation[3:6] = np.array([0,0,thetas[0]])
next_dtheta = dthetas[0]
next_d = dds[0]

n_start = 100
#n_end=4000
n_end=n_start+2000

df = data.values[n_start:n_end,:]

for i in range(1,n_end):
    theta = nthetas[i-1]
    dtheta = thetas[i]-theta
    dthetas[i] = dtheta
    ddx[i] = dx[i]-ndx[i-1]
    ddy[i] = dy[i]-ndy[i-1]
    
    """alpha = np.arctan(ddy[i]/ddx[i])
    print("dds",ddx[i],ddy[i],np.arctan(ddy[i]/ddx[i]),np.arctan2(ddy[i],0))
    
    print("alpha",alpha,theta,theta+dtheta/2,dtheta)"""
    
    d = np.linalg.norm(np.array([ddx[i],ddy[i],0]))
    #if d != 0:
    #    dtheta = 2*(alpha-theta)
    dds[i] = np.linalg.norm([ddx[i],ddy[i]])*np.sign(np.dot(np.array([d*mp.cos(theta+dtheta/2),d*mp.sin(theta+dtheta/2),mpf(0)]),np.array([ddx[i],ddy[i],0])))
    next_d = dds[i]
    d=next_d
    
    
    orientation[0:3] =orientation[0:3] +np.array([d*mp.cos(theta+dtheta/2),d*mp.sin(theta+dtheta/2),mpf(0)])#np.array(quat_rot(np.array([mpf(0),d*mp.cos(theta+dtheta/mpf(2)),d*mp.sin(theta+dtheta/mpf(2)),mpf(0)],dtype=mpf), quat),dtype=mpf)[1:4]
    orientation[3:6] = orientation[3:6]+np.array([0,0,mpf(dtheta)])
    
    next_dtheta = thetas[i]-orientation[5]
    
    ndx[i] = orientation[0]
    ndy[i] = orientation[1]
    nthetas[i] = orientation[5]
    
    
    
    
    
dleftw = dds-dthetas
drightw = dds+dthetas
    




odo=pd.read_csv(data_file,header=0)

odo_values = np.array([dleftw,drightw]).T

dtime = np.diff(time_odo)/10**9
arg_zeros = np.argwhere(dtime==0)
if len(arg_zeros)>0:
    zero_dtime = arg_zeros[0]
    dtime[zero_dtime]=1/10**10
v_odo_values = odo_values#np.diff(odo_values,axis=0)

velocity_odo = np.array([v_odo_values[i,:]/dtime[i] for i in range(len(dtime))])

imu_values = data.values[:,6:9]
time_imu = np.array([mp.mpmathify(data.values[i,5]) for i in range(len(data.values[:,5]))])



fs = 50
sos = butter(2, 24, fs=fs, output='sos')

smoothed = np.stack([
    sosfiltfilt(sos, velocity_odo[:, 0][n_start:n_end]),
    sosfiltfilt(sos, velocity_odo[:, 1][n_start:n_end]),
    sosfiltfilt(sos, np.interp(time_odo.astype(float),time_imu.astype(float),imu_values[:, 0].astype(float))[n_start:n_end]),
    sosfiltfilt(sos, np.interp(time_odo.astype(float),time_imu.astype(float),imu_values[:, 1].astype(float))[n_start:n_end]),
    sosfiltfilt(sos, np.interp(time_odo.astype(float),time_imu.astype(float),imu_values[:, 2].astype(float))[n_start:n_end])
], axis=1)
df[:,1:6] = smoothed

df[:,1:3] = velocity_odo[n_start:n_end,:]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[:,3:6])
g = 1.0
"""accx = df[:,5]*g
accy = -df[:,3]*g
accz = df[:,4] *g"""
accx = df[:,4]*g
accy = -df[:,3]*g
accz = df[:,5] *g


"""accx = df[:,4]*g
accy = df[:,3]*g
accz = -df[:,5] *g"""
df[:,3:6] = np.vstack([accx,accy,accz]).T

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[:,3:6])

quats = np.zeros((len(df[:,3:6]),4))
for i in range(len(df[:,3:6])):
    accx,accy,accz = df[i,3:6]
    p_acc = np.array([accx,accy,0])
    acc = np.array([accx,accy,accz])
    #qq = quat_ntom(acc/np.linalg.norm(acc), p_acc/np.linalg.norm(p_acc))
    qq = quat_ntom(acc/np.linalg.norm(acc), [1,0,0])
    quats[i,:] = (qq)
    #df[i,3:6] = np.array(quat_rot([0,*acc],qq))[1:4]
    
    
"""fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(df[:,3:6])"""
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quats)
q_calib = np.mean(quats[:100,:],axis=0)
q_calib = q_calib/np.linalg.norm(q_calib)
#normal0 = np.array(quat_rot([0,-1,0,0],(q_calib)))[1:4]
normal0 = np.array(quat_rot([0,1,0,0],quat_inv(q_calib)))[1:4]
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])
acc_mean = np.mean(df[:100,3:6],axis=0)
acc_angle = np.arctan2(acc_mean[1],acc_mean[0])#+np.arctan2(1,0)
acc_angle = np.arctan2(acc_mean[0],acc_mean[1])#+np.arctan2(1,0)

acc_reoriented = np.array(quat_rot([0,*acc_mean],ExpQua(zaxis*acc_angle)))[1:4]
normal0 = np.array(quat_rot([0,0,0,1],(quat_ntom(acc_reoriented,np.array([0,0,1])))))[1:4]

qq_final = quat_mult((quat_ntom(np.array([0,0,1]),normal0)),(ExpQua(zaxis*acc_angle)))

"""target_xaxis = np.array(quat_rot([0,*xaxis], ExpQua(zaxis*acc_angle)))[1:4]
qq0 = quat_ntom(zaxis, normal0)
#qqxy = np.array([1,0,0,0])
qq_final = quat_inv(quat_mult(qq0,ExpQua(zaxis*acc_angle)))
qq_test = (quat_rot(ExpQua(zaxis*acc_angle),qq0))"""
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
startleic = leica_values[:,3]
#df[:,6] = startleic
time= np.array(df[:,0],dtype=mpf)/10**9
df[0:,6] = np.interp(time.astype(float)-time[0]+3,time_leica.astype(float)-time_leica[0],startleic.astype(float))[0:n_end-n_start]-startleic[0]

#df[:,7:10] = c_mag
#newset = KFilterDataFile(df[:,:],mode=mmode,g_bias=g_bias,base_width=0.23,surf=np.array([-1,0,0.35])) 
#newset = KFilterDataFile(df[:,:],mode=mmode,base_width=0.23,surf=np.array([-1,0,0])) 
newset = KFilterDataFile(df[:,:],mode=mmode,base_width=2.0,surf=np.array(normal0))
#time= np.array(df[n_start:,0],dtype=mpf)/10**9

#newset = KFilterDataFile(df[:,:],mode=mmode,base_width=0.23,surf=np.array([-1,0,0])) 
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
q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(-2), 10**(-2)
q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(-1), 10**(-1)
#q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(0), 10**(0)
#normal = np.array([-1,1,0],dtype=mpf)10
normal = newset.normal



proj_func = correct_proj3
#proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(qq_final,dtype=mpf),proj_fun=None)
Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=np.array(qq_final,dtype=mpf),proj_fun=proj_func)#,grav=newset.grav)
Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(qq_final,dtype=mpf),proj_fun=proj_func,detection=True)#,grav=newset.grav)

ggg=Solv2.KFilter.gravity_r
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
    #print(Solv1.KFilter.position)
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

gravity_r = Solv2.gravity_r[:N,:]



for i in range(N):
    #print(quat_mult(quaternion[i,:],quat_inv(orient[i,:])))
    compare[i,:] = normalize(quat_mult(quaternion0[i,:],quat_inv(orient[i,:])))-np.array([1,0,0,0])
    compare2[i,:] = normalize(quat_mult(quaternion2[i,:],quat_inv(orient[i,:])))-np.array([1,0,0,0])
    #compare2[i,:] = quaternion2[i,:]-orient[i,:]

nn=int(N/2)    

fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
ax.plot([mp.norm(compare[i,:]) for i in range(len(compare))],'*')
ax.plot([mp.norm(compare2[i,:]) for i in range(len(compare2))],'*')
ax.set_title('angle = ' + str(angle) + ' rad, bias =' +str(g_bias) + ', gyro noise =' + str(g_noise) + ', acc noise =' + str(a_noise))


#print(list(map(surf,position2)))

diff_pos1 = [np.linalg.norm(a) for a in position1-pos_earth]
diff_pos2 = [np.linalg.norm(a) for a in position2-pos_earth]


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




    
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc[:,0]/newset.acc[:,2])




#leic = df[:,1:4]
leic = leica_values[:,1:4]

rel_df = leic[:,:]-leic[0,:]

for i in range(len(rel_df)):
    lei = rel_df[i,:]
    p_lei = np.array([0,-lei[1],lei[2]])
    if np.linalg.norm(p_lei) > 0:
        qq = quat_ntom(lei/np.linalg.norm(lei), p_lei/np.linalg.norm(p_lei))
        rel_df[i,:] = np.array(quat_rot([0,*lei],qq))[1:4]
        rel_df[i,0]=-rel_df[i,1]
        rel_df[i,1]=0


leic[:,:] = rel_df



#df[:,6] = startleic
leic_compare = np.array([np.interp(time.astype(float)-time[0]-2,time_leica.astype(float)-time_leica[n_start],leic[:,0].astype(float))[:n_end-n_start],
                         np.interp(time.astype(float)-time[0]-2,time_leica.astype(float)-time_leica[n_start],leic[:,1].astype(float))[:n_end-n_start],
                         np.interp(time.astype(float)-time[0]-2,time_leica.astype(float)-time_leica[n_start],leic[:,2].astype(float))[:n_end-n_start]]).T


position3= np.copy(position2)

for i in range(len(position2)):
    lei = position2[i,:]
    p_lei = np.array([0,lei[1],lei[2]])
    if np.linalg.norm(p_lei) > 0:
        qq = quat_ntom(lei/np.linalg.norm(lei), p_lei/np.linalg.norm(p_lei))
        position3[i,:] = np.array(quat_rot([0,*lei],qq))[1:4]

nposition0 = -np.array([np.array(quat_rot([0,*p], (quat_ntom(normal0,yaxis))))[1:4] for p in position0])
nposition1 = -np.array([np.array(quat_rot([0,*p], (quat_ntom(normal0,yaxis))))[1:4] for p in position1])
nposition2 = -np.array([np.array(quat_rot([0,*p], (quat_ntom(normal0,yaxis))))[1:4] for p in position2])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time[:n_end-n_start]-time[0],-nposition0[:,0])
ax.plot(time[:n_end-n_start]-time[0],-nposition1[:,0])
ax.plot(time[:n_end-n_start]-time[0],-nposition2[:,0])
ax.plot(time[:n_end-n_start]-time[0],leic_compare[:,0])
ax.set_title('Position of Integration of Odometry x')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time[:n_end-n_start]-time[0],nposition0[:,1])
ax.plot(time[:n_end-n_start]-time[0],nposition1[:,1])
ax.plot(time[:n_end-n_start]-time[0],nposition2[:,1])
ax.plot(time[:n_end-n_start]-time[0],leic_compare[:,1])
ax.set_title('Position of Integration of Odometry y')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time[:n_end-n_start]-time[0],nposition0[:,2])
ax.plot(time[:n_end-n_start]-time[0],nposition1[:,2])
ax.plot(time[:n_end-n_start]-time[0],nposition2[:,2])
ax.plot(time[:n_end-n_start]-time[0],leic_compare[:,2])
ax.set_title('Position of Integration of Odometry z')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(nposition1[:,2],nposition1[:,0])
ax.plot(nposition2[:,2],nposition2[:,0])
ax.plot(leic_compare[:,2],-leic_compare[:,0]+0.27-0.05)
ax.legend(['MEKF','Rev-MEKF','Leica'])
ax.set_title('Comparison of trajectories on a plane')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(nposition2[:,2],nposition2[:,0])
ax.set_title('Position with Reversible MEKF')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(nposition1[:,2],nposition1[:,0])
ax.set_title('Position with MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(gravity_r[1:,:])
ax.set_title('Gravity obtained from Rev-MEKF')

normed_acc = np.array([a/np.linalg.norm(a) for a in newset.acc])
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(normed_acc)
ax.set_title('Gravity obtained from MEKF')