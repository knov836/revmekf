import numpy as np
import sympy as sym
import mpmath as mp
from mpmath import mpf
import numdifftools as nd 
from function_quat import *

def correct_proj3(dt,Quaternion,Pk,position,center,Accelerometer,Surface0,Surface1,grav=1):
    normal = Surface0[1:4]
    normal = normal/mp.norm(normal)
    
    normal1 = Surface1[1:4]
    normal1 = normal1/mp.norm(normal1)
    
    ax,ay,az=Accelerometer
    acc_earth = np.array(quat_rot([0,ax,ay,az],Quaternion)[1:4],dtype=mpf)
    
    '''step = acc_earth*(dt)**2
    cor_racc = (position+step).dot(normal1)
    acc_earth = (step-cor_racc*normal1)/(dt**2)'''
    step = acc_earth*(dt)**2
    
    cor_racc = (position+step-center).dot(normal)
    #print("======================================================")
    #print("1",acc_earth)
    acc_earth = (step-cor_racc*normal)/(dt**2)
    #print("2",acc_earth)
    step = acc_earth*(dt)**2
    
    return Quaternion,Pk,acc_earth#*coef
    

def correct_proj2(dt,Quaternion,Pk,position,center,Accelerometer,Surface,grav=1):
    normal = Surface[1:4]
    normal = normal/mp.norm(normal)
    ax,ay,az=Accelerometer
    acc_earth = np.array(quat_rot([0,ax,ay,az],Quaternion)[1:4],dtype=mpf)
    
    step = acc_earth*(dt)**2
    coef = mpf(1)
    #coef = mpf(-np.dot(position,normal)/np.dot(step,normal))
    
    cor_racc = (position+step -center).dot(normal)
    acc_earth = (step-cor_racc*normal)/(dt**2)
    step = acc_earth*(dt)**2
    
    n_pos = position+coef*step
    #print("n_pos",np.dot(n_pos,normal),center)
    return Quaternion,Pk,acc_earth#*coef
    
    

def correct_proj1(dt,Quaternion,Pk,position,center,Accelerometer,Surface,grav=-1):
    ax,ay,az = Accelerometer
    #g=mpf(-1)
    normal = Surface[1:4]
    normal = normal/mp.norm(normal)
    
    Quat = Quaternion
    
    acc_earth = np.array(quat_rot([0,ax,ay,az],Quaternion)[1:4],dtype=mpf)
    
    dvec = acc_earth*(dt)**2
    
    D = mp.norm(dvec)
    l = position-center
    #print("pos center",position, center,dvec)
    
    if np.dot(l,l)<=D**2:
        R = np.sqrt(D**2-np.dot(l,l))
    
    #R = np.sqrt(D**2-np.dot(l,l))
    #print("R",R,"D",D**2,"l",np.dot(l,l))
    #print("D",D**2,"l",np.dot(l,l))
    #print(np.dot(l,l)-D**2)
    if np.dot(l,l)>D**2:
        
        aex0,aey0,aez0 = acc_earth
        acc_earth = (position-center)/(dt**2)*grav
        aex,aey,aez = acc_earth
        #print(aex0,aex,aey,aey0,aez,aez0)
        #print("change orientation of acc")
        #(Accelerometer)
        Accelerometer = np.array(quat_rot([0,aex,aey,aez],quat_inv(Quaternion))[1:4],dtype=mpf)
        #print(Accelerometer)
        #print(np.array(quat_rot([0,aex0,aey0,aez0],quat_inv(Quaternion))[1:4],dtype=mpf),Accelerometer)
        ax,ay,az=Accelerometer
        dvec = acc_earth*(dt)**2
        
        D = l
        R = 0
        

    
    
    
    acc = Accelerometer*(dt)**2
    
    
    a_prj =  position+dvec-np.dot(position+dvec,normal)*normal+np.dot(center,normal)*normal

    r = mp.norm(center-a_prj)
    
    if r==0:
        a_prj = center
    else:
        a_prj= center + (a_prj-center)/r*R #on the circle centered at center
        
    
    #print("R",R,"D",D**2,"l",np.dot(l,l))
    
    h_prj = a_prj-position
    #print("origin",dvec,"target",h_prj)
    #nh_prj = 
    state = log_q(Quaternion)
    Q = ExpQua(state)
    q_acc = np.array([0,*acc],dtype=mpf)
    
    nplan0 = h_prj-acc
    nplan1 = dvec-acc
    #print("nplan0",nplan0)
    #print("nplan1",nplan1)
    #print("in plan 0",np.linalg.norm(nplan0)< 1e-40)
    #print(np.linalg.norm(nplan0))
    #print(nplan0,nplan1)
    #print(normal)
    if (np.linalg.norm(nplan0)< 1e-40):
        nplan0 = normal
        nplan1 = normal
    #nrot = skewSymmetric(nplan1/mp.norm(nplan1))@nplan0/mp.norm(nplan0)
    
    Transition = np.eye(6,dtype=mpf)
    #beta = mp.asin(mp.norm(nrot))
    
    #qrot = ExpQua(nrot/mp.norm(nrot)*beta)
    
    qrot = quat_ntom(nplan1, nplan0)
    Transition[:3,:3] = QuatToRot(qrot)
    rstate = np.array(quat_rot([0,*state],qrot)[1:4])
    
    ffQ = ExpQua(rstate)
    
    nrstate = rstate
    if(mp.norm(rstate) != 0):
        nrstate = rstate/mp.norm(rstate)
    
    
    pacc = acc -np.dot(nrstate,acc)*nrstate
    ph_prj = h_prj -np.dot(nrstate,h_prj)*nrstate
    #print("pacc ph_prj",pacc,ph_prj)
    mm = np.dot((pacc/mp.norm(pacc)),ph_prj/mp.norm(ph_prj))
    
    alpha = mp.acos(mm/np.max([1,np.abs(mm)]))
    
    #nplan2 = skewSymmetric(pacc/mp.norm(pacc))@ph_prj/mp.norm(ph_prj)
    #print(alpha)
    
    coef = mpf(1)
    if(mp.norm(rstate) != 0):
        coef = mpf(1)/mp.norm(rstate)
    rstate  = nrstate*alpha
    coef *=mpf(alpha)
    Transition[:3,:3] *= coef
    qq = quat_ntom(pacc,ph_prj)
    fQ = ExpQua(rstate)
        
    
    #self.Quaternion = np.copy(Quat)
    Quaternion = np.array(fQ,dtype=mpf)
    Pk = Transition@Pk@Transition.T
    oacc_earth = np.copy(acc_earth)
    acc_earth = np.array(quat_rot([0,ax,ay,az],Quaternion)[1:4],dtype=mpf)
    #print("compare acc_earth",oacc_earth,acc_earth)    
    return Quaternion,Pk,acc_earth


def correct_proj0(dt,Quaternion,Pk,position,center,Accelerometer,Surface):
    Quat=np.copy(Quaternion)
    normal = np.array([mpf(s) for s in Surface[1:4]],dtype=mpf)
    normal1 = np.array(quat_rot([0,0,0,1], quat_inv(Quaternion))[1:4],dtype=mpf) 
    
    n0,n1 =mp.norm(normal),mp.norm(normal1)
    normal = normal/n0
    newquat = skewSymmetric(normal1)@normal/(n1)
    alpha = mp.asin(mp.norm(newquat))
    if alpha!=0:
        newquat = newquat/mp.sin(alpha)*mp.sin(alpha/2)
    else:
        newquat = newquat*0
    Quaternion = np.zeros(4,dtype=mpf)
    Quaternion[1:4] = newquat
    Quaternion[0] = mp.cos(alpha/2)
    Quaternion = np.array(quat_mult(Quat,quat_inv(Quaternion)))
    
    
    return Quaternion,Pk

def correct_pos0(position,Pk,Surface):
    print(Surface,position)
    surf = lambda x : float(Surface[0]) + Surface[1]*x[0]+ Surface[2]*x[1]+ Surface[3]*x[2]+ Surface[4]*x[0]*x[1]+ Surface[5]*x[1]*x[2]+ Surface[6]*x[2]*x[0] + Surface[7]*x[0]**2+ Surface[8]*x[1]**2+ Surface[9]*x[2]**2
    grad = nd.Gradient(surf) #
    position = project(position, surf,grad)
    return position,Pk