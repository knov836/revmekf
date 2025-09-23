
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt

from variants.integration_gyroaccmag import PredictFilter as IntegrationGyroAccMag
from variants.integration_gyroacc import PredictFilter as IntegrationGyroAcc
from variants.mekf_gyroaccmag import PredictFilter as MEKFGyroAccMag

from variants.mekf_gyroacc import PredictFilter as MEKFGyroAcc
from variants.reversible_gyroaccmag import PredictFilter as RevGyroAccMag
from variants.reversible_gyroacc import PredictFilter as RevGyroAcc

from variants.integration_odoaccpre import PredictFilter as IntegrationOdoAccPre
from variants.mekf_odoaccpre import PredictFilter as MEKFOdoAccPre
from variants.reversible_odoaccpre import PredictFilter as RevOdoAccPre

from mpmath import mp
from mpmath import mpf
from proj_func import correct_proj2
from function_quat import *
from predict_odo_test import test_predict
from update_odo_test import test_update


from kfilterdata_synthetic import KFilterDataSynth as KFilterData


mp.dps = 40

import sys,os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split('/examples')[0])


from scipy.spatial.transform import Rotation
from solver_kalman import SolverFilterPlan

N=100
nn = N-1
g_bias= 10**(-5)
g_noise=10**(-10)
a_noise=10**(-4)
angle = int(N/2)
#mmode = 'OdoAccPre'
mmode = 'GyroAccMag'
#mmode = 'OdoAccPre'

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
delt = 60
newset = KFilterData(100,mpf(100.)/mpf(1.),mode=mmode,traj='Rand',lw_noise=0.1*0,rw_noise=0.1*0,g_bias= 10**(-3)*0,g_noise=10**(-10)*0,a_noise=10**(-4)*0,params_test={'alpha':angle},surf=np.array([0,1,-1],dtype=mpf),delta = 10**(-delt*0.1)) 
orient = newset.orient
pos_earth = newset.pos_earth

q0,q1,r0,r1 = 10**(-5), 10**(-5), 10**(8), 10**(-2) 
q0,q1,r0,r1 = 10**(-5), 10**(-5), 10**(-10), 10**(-10)
q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(-2), 10**(-2)
#normal = np.array([-1,1,0],dtype=mpf)
normal = newset.normal
proj_func = correct_proj2
proj_func = None
if mmode.startswith('Gyro'):
    Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=newset.orient[0,:],proj_fun=proj_func )
    Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=newset.orient[0,:],proj_fun=proj_func )#,grav=newset.grav)
    Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=newset.orient[0,:],proj_fun=proj_func )#,grav=newset.grav)
    
if mmode.startswith('Odo'):
    """Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,proj_fun=proj_func)
    Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,proj_fun=proj_func)#,grav=newset.grav)
    Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,proj_fun=proj_func)#,grav=newset.grav)"""
    
    Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=newset.orient[0,:].flatten())
    Solv1 = SolverFilterPlan(MEKF,q0,q1,r0,r1,normal,newset,start=newset.orient[0,:].flatten())#,grav=newset.grav)
    Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=newset.orient[0,:].flatten())#,grav=newset.grav)
#Solv2 = SolverFilterPlan(PredictFilterPlan,q0,q1,r0,r1,normal,None)   

N=newset.size
nn=0
neworient = newset.new_orient()
#N=2
for i in range(0,N-1,1):
    nn+=1
    Solv0.update_noarg()
    Solv1.update_noarg()
    Solv2.update_noarg()
    
    '''if i%10 ==0 and i>0:
        """fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(Solv1.position[i-10:i,:])
        ax.set_title('position1')"""
    
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot([np.linalg.norm(Solv1.position[i-10+j,:]) for j in range(10)])
        ax.plot([np.linalg.norm(Solv2.position[i-10+j,:]) for j in range(10)])
        
        plt.show()'''



 
compare = np.zeros((N,4),dtype=float)
compare2 = np.zeros((N,4),dtype=float)
quaternion0 = Solv0.quaternion[:N,:]    
quaternion1 = Solv1.quaternion[:N,:]
quaternion2 = Solv2.quaternion[:N,:]
#quaternion2 = np.copy(quaternion0)
position0 = Solv0.position[:N,:]
position1 = Solv1.position[:N,:]
position2 = Solv2.position[:N,:]
#position2 = np.copy(position0)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion0[:,0],quaternion0[:,1],'*')
ax.plot(quaternion2[:,0],quaternion2[:,1],'*')

ax.legend(['Input quaternion','Resulting quaternion'])
ax.set_title('Comparison of initial quaternion and result with freq=' + str(newset.freq))


for i in range(N):
    compare[i,:] = normalize(quat_mult(quaternion0[i,:],quat_inv(orient[i,:])))-np.array([1,0,0,0])
    compare2[i,:] = normalize(quat_mult(quaternion2[i,:],quat_inv(orient[i,:])))-np.array([1,0,0,0])

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
ax.set_title('Orientation from Integration of Gyroscope')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion1)
ax.set_title('Orientation from MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(quaternion2)
ax.set_title('Orientation from Rev-MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.orient)
ax.set_title('True Orientation')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0)
ax.set_title('Position from Integration of Gyroscope')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position1)
ax.set_title('Position from MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2)
ax.set_title('Position from Rev_MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth)
ax.set_title('Position in Earth Frame')



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position0)
ax.set_title('Comparison of position of Integration of Gyroscope')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position1)
ax.set_title('Comparison of position of MEKF')

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


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position0)
ax.set_title('Position from Integration of Gyroscope')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position1)
ax.set_title('Position from MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2)
ax.set_title('Position from Rev_MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth)
ax.set_title('Position in Earth Frame')



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position0)
ax.set_title('Comparison of position of Integration of Gyroscope')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position1)
ax.set_title('Comparison of position of MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(pos_earth-position2)
ax.set_title('Comparison of position of Rev-MEKF')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare0)
ax.set_title('Comparison with integration of Odometry')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare1)
ax.set_title('Comparison with MEKF')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(compare2)
ax.set_title('Comparison with Rev-MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(newset.acc)
ax.set_title('Acceleration for 10^(-'+str(delt*0.1)+')')

