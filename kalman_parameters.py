import numpy as np
from mpmath import mpf

def generate_kalman_parameters(q0,q1,r0,r1):
    qq0= q0
    qq1 = q1

    
    RR=np.eye(6,dtype=mpf)

    QQ=np.eye(6,dtype=mpf)*qq0
    for i in range(3,6):
        QQ[i,i] = qq1
        
    PK=np.eye(6)
    for i in range(3):
        RR[i,i] = r0
        
    for i in range(3,6):
        RR[i,i] = r1
    return QQ,RR,PK

def generate_kalman_parameters_acconly(q0,q1,r0):
    qq0= q0
    qq1 = q1

    
    RR=np.eye(3,dtype=mpf)

    QQ=np.eye(6,dtype=mpf)*qq0
    for i in range(3,6):
        QQ[i,i] = qq1
        
    PK=np.eye(6)
    for i in range(3):
        RR[i,i] = r0
        
    return QQ,RR,PK


def generate_kalman_parameters_odo(q0,q1,r0,r1):
    qq0= q0
    qq1 = q1

    
    RR=np.eye(3,dtype=mpf)

    QQ=np.eye(2,dtype=mpf)*qq0
    PK=np.eye(6)
    for i in range(3):
        RR[i,i] = r0
    return QQ,RR,PK
