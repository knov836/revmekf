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
