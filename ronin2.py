import pandas as pd

# Using numpy quaternions for rotation manipulation.
import numpy as np
#import quaternion

# Using matplolib for visualization.
import matplotlib.pyplot as plt
import matplotlib as mpl
from function_quat import *

# Using opencv to parse the video file.

def my_procrustes(m1,m2):
    #procrustes algorithm with no scaling and no reflection.    

    mu1=np.mean(m1,0)
    mu2=np.mean(m2,0)
    
    m10=m1-mu1
    m20=m2-mu2
    
    ssq1=np.sum(m10**2,0)
    ssq2=np.sum(m20**2,0)
    
    ssq1=np.sum(ssq1)
    ssq2=np.sum(ssq2)
    
    n1=np.sqrt(ssq1)
    n2=np.sqrt(ssq2)
 
    m10=m10/n1
    m20=m20/n2
    
    A=m10.T@m20
    L,d,M=np.linalg.svd(A)
    M=M.T
    T=M@L.T
 
    if np.linalg.det(T)<0:
        M[:,-1]=-M[:,-1]
        d[-1]=d[-1]
        T=M@L.T
    
    trac=sum(d)
    
    scale=1
    d=1 + ssq2/ssq1 - 2*trac*n2/n1
    
    trans=mu1-scale*mu2@T
    return T,trans
    
def align_to_fixpoints(M_path,M_fix):
    # align pose track to fixpoints.
    samp_index=[]
    for i in range(0,np.shape(M_fix)[0]):
        samp_index.append(int(np.argmin((M_path[:,0]-M_fix[i,0])**2)))
    R,trans=my_procrustes(M_fix[:,1:4],M_path[samp_index,1:4])
    trans= np.array(trans)[np.newaxis]
    M_path2=M_path
    M_path2[:,1:4]=(M_path[:,1:4]@R+trans)
    for i in range(0,np.shape(M_path)[0]):
        quat=np.array((M_path[i,4],M_path[i,5],M_path[i,6],M_path[i,7]))
        R_q=QuatToRot(quat)
        n_quat=RotToQuat(R.T@R_q)
        M_path2[i,4:8]=np.array(n_quat)   
    return(M_path2)
def fit_linear_R(a, b):
    a = np.asarray(a).astype(float)
    b = np.asarray(b).astype(float)
    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if b.ndim == 2 and b.shape[0] == 3 and b.shape[1] != 3:
        b = b.T

    A = a.T
    B = b.T

    H = A @ B.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # fix reflection if it occurred (ensure det = +1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    return R


def sync_a(x,y):
    score = 0
    min = 0
    for t0 in range(0,200):
        z = np.copy(np.roll(y,t0*10))
        norm = np.sum(np.abs(x-z))/len(x)
        print(norm)
        if norm>min:
            min = norm
            score=t0
    return score

def fitEllipsoid(magX, magY, magZ):
    a1 = magX ** 2
    a2 = magY ** 2
    a3 = magZ ** 2
    a4 = 2 * np.multiply(magY, magZ)
    a5 = 2 * np.multiply(magX, magZ)
    a6 = 2 * np.multiply(magX, magY)
    a7 = 2 * magX
    a8 = 2 * magY
    a9 = 2 * magZ
    a10 = np.ones(len(magX)).T
    D = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])

    # Eqn 7, k = 4
    C1 = np.array([[-1, 1, 1, 0, 0, 0],
                   [1, -1, 1, 0, 0, 0],
                   [1, 1, -1, 0, 0, 0],
                   [0, 0, 0, -4, 0, 0],
                   [0, 0, 0, 0, -4, 0],
                   [0, 0, 0, 0, 0, -4]])

    # Eqn 11
    S = np.matmul(D, D.T)
    S11 = S[:6, :6]
    S12 = S[:6, 6:]
    S21 = S[6:, :6]
    S22 = S[6:, 6:]

    # Eqn 15, find eigenvalue and vector
    # Since S is symmetric, S12.T = S21
    tmp = np.matmul(np.linalg.inv(C1), S11 - np.matmul(S12, np.matmul(np.linalg.inv(S22), S21)))
    eigenValue, eigenVector = np.linalg.eig(tmp)
    u1 = eigenVector[:, np.argmax(eigenValue)]

    # Eqn 13 solution
    u2 = np.matmul(-np.matmul(np.linalg.inv(S22), S21), u1)

    # Total solution
    u = np.concatenate([u1, u2]).T

    Q = np.array([[u[0], u[5], u[4]],
                  [u[5], u[1], u[3]],
                  [u[4], u[3], u[2]]])

    n = np.array([[u[6]],
                  [u[7]],
                  [u[8]]])

    d = u[9]

    return Q, n, d

my_top_percentile = 0.025 
my_bottom_percentile = 0.025
save_plots_to_file = True   
plot_sampling_percentage = 0.2 
def filter_outlier(df, top_percentile=my_top_percentile, bottom_percentile=my_bottom_percentile):
    # Filter out the top 5% and bottom 5% values in each column to remove noise
    for column in ['x', 'y', 'z']:
        lower_bound = df[column].quantile(0 + bottom_percentile/2)
        upper_bound = df[column].quantile(1 - top_percentile/2)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def apply_offset_correction(df):
    offset_x = (max(df['x']) + min(df['x'])) / 2
    offset_y = (max(df['y']) + min(df['y'])) / 2
    offset_z = (max(df['z']) + min(df['z'])) / 2
    corrected_df = pd.DataFrame({
        'corrected_x': df['x'] - offset_x,
        'corrected_y': df['y'] - offset_y,
        'corrected_z': df['z'] - offset_z
    })
    return offset_x, offset_y, offset_z, corrected_df

def apply_scale_correction(df):
    avg_delta_x = (max(df['corrected_x']) - min(df['corrected_x'])) / 2
    avg_delta_y = (max(df['corrected_y']) - min(df['corrected_y'])) / 2
    avg_delta_z = (max(df['corrected_z']) - min(df['corrected_z'])) / 2
    avg_delta = (avg_delta_x + avg_delta_y + avg_delta_z) / 3
    scale_x = avg_delta / avg_delta_x
    scale_y = avg_delta / avg_delta_y
    scale_z = avg_delta / avg_delta_z
    scaled_corrected_df = pd.DataFrame({
        'scaled_corrected_x': df['corrected_x'] * scale_x,
        'scaled_corrected_y': df['corrected_y'] * scale_y,
        'scaled_corrected_z': df['corrected_z'] * scale_z
    })
    return scale_x, scale_y, scale_z, scaled_corrected_df