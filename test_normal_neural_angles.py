import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from datetime import datetime
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
import glob

import torch
import torch.nn as nn
import torch.optim as optim
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='tanh', dropout=0.1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # out: (batch, timesteps, hidden)
        out, hn = self.rnn(x)       # hn = dernier état caché (num_layers, batch, hidden)
        out = self.dropout(hn[-1])  # on prend le dernier état caché de la dernière couche
        out = self.fc(out)
        return out
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # out: (batch, timesteps, hidden)
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(hn[-1])  # dernier état caché
        out = self.fc(out)
        return out

#model = torch.load("lstm_model_20251012_105842.pth",weights_only=False)
#good model
#model = torch.load("lstm_model_20251012_173153.pth",weights_only=False)

model = torch.load("lstm_model_20251012_173153.pth",weights_only=False)
#model = torch.load("rnn_model.pth",weights_only=False)
model.eval()
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
n_end=n_start +3000
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

q0,q1,r0,r1 = 10**(-2), 10**(-2), 10**(0), 10**(0)
normal = newset.normal


gravity = newset.gravity



proj_func = correct_proj2
proj_func = None
Solv0 = SolverFilterPlan(Integration,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func)
Solv1 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True)#,grav=newset.grav)
Solv2 = SolverFilterPlan(Rev,q0,q1,r0,r1,normal,newset,start=np.array(newset.quat_calib,dtype=mpf),proj_fun=proj_func,heuristic=True,neural=True)#,grav=newset.grav)

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

normals = compute_normals(N,acc_smooth,gravity,df[:,7:10])
y = normals
y_smooth0 = savgol_filter(y[:,0], 500, 2)
y_smooth1 = savgol_filter(y[:,1], 500, 2)
y_smooth2 = savgol_filter(y[:,2], 500, 2)
y_smooth = np.vstack((y_smooth0,y_smooth1,y_smooth2)).T
normals = np.copy(y_smooth)


rows = []


window = 20
time0=time-time[0]
for p1 in range(window,N,1):
    p0 = p1
    p_start = max(0, p0 - window)
    p_end   = min(N, p0 + window)
    indices = list(range(p_start, p0))  # indices du voisinage
    
    
    

    row = {
        "sample": int(p0)+n_start,
        "time": float(time0[p0]+time[0]),
    }

    for k, j in enumerate(indices):
        row[f"t0_{k}"] = float(0)
        row[f"t2_{k}"] = float(0)
        row[f"t3_{k}"] = float(0)
        row[f"t4_{k}"] = float(0)
        
        row[f"et0_{k}"] = float(0)
        row[f"et2_{k}"] = float(0)
        row[f"et3_{k}"] = float(0)
        row[f"et4_{k}"] = float(0)
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

"""df.to_csv(f"corrections_windows_tmp.csv", index=False)
files = glob.glob("corrections_windows_tmp.csv")"""
#df_list = [pd.read_csv(f) for f in files]
#df = pd.concat(df_list, ignore_index=True)
df.interpolate(method='linear', axis=0, inplace=True)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

blocks = ["normal","acc", "gyro", "mag"]
angles = ["t0","t2", "t3", "t4","et0","et2", "et3", "et4",]

axes = ["x", "y", "z"]
timesteps = 20  

seq_features = []
a_seq_features = []
extra_features=[]
for block in blocks:
    for axis in axes:
        seq_features += [f"{block}_{axis}_{i}" for i in range(timesteps)]
for angle in angles:
        a_seq_features += [f"{angle}_{i}" for i in range(timesteps)]

for block in blocks:
    
    for axis in axes:
        cols = [c for c in df.columns if c.startswith(f"{block}_{axis}_")]
        print(df[cols])
        print(cols)
        smoothed_cols = savgol_filter(np.array(df[cols]), 5, 2)
        df[f"{block}_{axis}_mean"] = df[cols].mean(axis=1)
        df[f"{block}_{axis}_std"] = df[cols].std(axis=1)
        df[f"{block}_{axis}_min"] = df[cols].min(axis=1)
        df[f"{block}_{axis}_max"] = df[cols].max(axis=1)
        """feature_cols += [f"{block}_{axis}_mean", f"{block}_{axis}_std",
                         f"{block}_{axis}_min", f"{block}_{axis}_max"]
        """
        extra_features += [f"{block}_{axis}_std"]
    # Norme of vector
    if block == "normal":
        continue
    normal = df[[f"normal_{ax}_mean" for ax in axes]]
    df[f"{block}_norm"] = np.sqrt(df[[f"{block}_{ax}_mean" for ax in axes]].pow(2).sum(axis=1))
    df[f"{block}_norm_crossnormal"] = np.sqrt((np.cross(normal,df[[f"{block}_{ax}_mean" for ax in axes]])**2).sum(axis=1))
    
    extra_features+= [f"{block}_norm_crossnormal"]
"""for block in angles:
    cols = [c for c in df.columns if c.startswith(f"{block}_")]
    df[f"{block}_mean"] = df[cols].mean(axis=1)
    df[f"{block}_std"] = df[cols].std(axis=1)
    df[f"{block}_min"] = df[cols].min(axis=1)
    df[f"{block}_max"] = df[cols].max(axis=1)
    extra_features += [f"{block}_mean", f"{block}_std",
                     f"{block}_min", f"{block}_max"]"""


#seq_features += [f"normal_{a}" for a in {"z"} for i in range(timesteps)]
seq_features = seq_features+extra_features
input_dim = 12+8+len(extra_features)
#X = df[seq_features].values.reshape(len(df), timesteps, input_dim)

X_seq = df[[f for f in seq_features if f not in extra_features]].values
X_extra = df[extra_features].values  # shape = (n_samples, n_extra)

X_extra_seq = np.repeat(X_extra[:, np.newaxis, :], timesteps, axis=1)

"""X = np.concatenate([X_seq.reshape(len(df), timesteps, 12+8), X_extra_seq], axis=2)
X_tensor = torch.tensor(X, dtype=torch.float32)"""
preds_threshold = np.zeros(N)
probs = np.zeros(N)
threshold=0.82
threshold=0.52
threshold=0.5
threshold=0.8

angles_array = np.zeros((len(angles),window))
angle_features = np.array([])

angle_means = np.mean(angles_array, axis=1)  # (len(angles),)
angle_stds  = np.std(angles_array, axis=1)
angle_mins  = np.min(angles_array, axis=1)
angle_maxs  = np.max(angles_array, axis=1)

coords1 = np.zeros(coords.shape)
#coords1[:,0] = (coords[:,0]+coords[:,1])/np.sqrt(2)
#coords1[:,1] = (coords[:,0]-coords[:,1])/np.sqrt(2)
coords1 = coords
with torch.no_grad():
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
        correction = 0
        X_input = np.array([])
        if i+1>window:
            
            #extra_features = [f"{block}_{stat}" for block in angles for stat in ["mean", "std", "min", "max"]]
            X_extra = df.loc[i+1-window, extra_features].values[np.newaxis, :] 
            X_extra_seq = np.repeat(X_extra[:, np.newaxis, :], timesteps, axis=1)
            X_seq_i = X_seq[i+1-window].reshape(1, timesteps, 12)
    
            angles_seq = np.repeat(angles_array[np.newaxis, :, :], 1, axis=0)
            angles_seq = np.transpose(angles_seq, (0, 2, 1))  
    
            
            X_extra_angles = angle_features[np.newaxis, :]
            X_extra_angles_3D = np.repeat(X_extra_angles[:, np.newaxis, :], timesteps, axis=1)
            X_input = np.concatenate([X_seq_i, angles_seq, X_extra_seq,X_extra_angles_3D], axis=2)
    
            """X_tensor_1 = torch.tensor(X_input, dtype=torch.float32)
            outputs = model(X_tensor_1)
            soft = torch.softmax(outputs, dim=1).numpy().squeeze()
    
            p_class0 = soft[0]
            p_class1 = soft[1]
            probs[i] = p_class1
            correction = int(p_class1 > threshold)
            
            print(p_class0,p_class1,correction)"""
        
        Solv0.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
        Solv1.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal)
        Solv2.update(time[i+1], newset.gyro[i+1,:], newset.acc[i+1,:], newset.mag[i+1,:], normal,correction=correction,model=model,xtensor=X_input)
        
        current_angles = np.array([Solv2.KFilter.t0,Solv2.KFilter.t2,Solv2.KFilter.t3,Solv2.KFilter.t4,Solv2.KFilter.et0,Solv2.KFilter.et2,Solv2.KFilter.et3,Solv2.KFilter.et4])
        #print(angles_array)
        angles_array = np.roll(angles_array, shift=-1, axis=1)
        #print(angles_array)
        angles_array[:, -1] = current_angles  
        #print(current_angles)
        angle_means = np.mean(angles_array[:,1:], axis=1)  # (len(angles),)
        angle_stds  = np.std(angles_array[:,1:], axis=1)
        angle_mins  = np.min(angles_array[:,1:], axis=1)
        angle_maxs  = np.max(angles_array[:,1:], axis=1)
        angle_features = np.concatenate([angle_means, angle_stds, angle_mins, angle_maxs])
        
        correction_applied[i] = Solv2.KFilter.corrected
        angle_applied[i+1] =angle_applied[i]+Solv2.KFilter.angle
    
        if i%10 ==0 and i>0:
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            ax.plot([np.linalg.norm(Solv0.position[j,:]) for j in range(i+1)])
            ax.plot([np.linalg.norm(Solv1.position[j,:]) for j in range(i+1)])
            ax.plot([np.linalg.norm(Solv2.position[j,:]) for j in range(i+1)])        
            ax.plot([np.linalg.norm(coords[j,:]) for j in range(i+1)])
            
            plt.show()
            
            
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            ax.plot([Solv0.position[j,:2] for j in range(i+1)])
            ax.plot([Solv1.position[j,:2] for j in range(i+1)])
            ax.plot([Solv2.position[j,:2] for j in range(i+1)])
            ax.plot(np.argwhere(correction_applied).flatten(), [Solv2.position[j,:2]  for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))
            ax.plot([coords1[j,:] for j in range(i+1)])
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


size  =n_end-n_start

#size = n_start+N



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(time0[:size-1],[np.linalg.norm(Solv0.position[j,:]) for j in range(size-1)])
ax.plot(time0[:size-1],[np.linalg.norm(Solv1.position[j,:]) for j in range(size-1)])
ax.plot(time0[:size-1],[np.linalg.norm(Solv2.position[j,:]) for j in range(size-1)])
ax.plot(time0[:size-1],[np.linalg.norm(coords[j,:]) for j in range(size-1)])
ax.plot(time0[np.argwhere(correction_applied).flatten()], [np.linalg.norm(Solv2.position[j,:]) for j in np.argwhere(correction_applied).flatten()],'.',**dict(markersize=10))

ax.legend(['Integration of Gyroscope','Heuristic Rev-MEKF','LSTM Rev-MEKF','GPS','Corrections applied by Rev-MEKF'])
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
    q0[i,:] = np.array(quat_rot([0,*newset.mag[i,:]],(quaternion0[i,:])))[1:4]
    q1[i,:] = np.array(quat_rot([0,*newset.mag[i,:]],(quaternion1[i,:])))[1:4]
    q2[i,:] = np.array(quat_rot([0,*newset.mag[i,:]],(newset.neworient[i,:])))[1:4]
    
    
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
ax.set_title('Position from Heuristic Rev-MEKF')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(position2[:,:2])
ax.plot(np.array(coords1))
ax.legend(['pos0_x','pos0_y','gps_x,','gps_y'])
ax.set_title('Position from LSTM Rev-MEKF')
