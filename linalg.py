
import numpy as np
from math import pi
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import sympy
from mpmath import mpf
from mpmath import mp
import numdifftools as nd 

from sympy.plotting import plot

import sympy as sym


from sympy import symbols, cos, sin
from sympy import Rational
from sympy import nsolve
from sympy import lambdify
from sympy import series
from sympy.plotting import plot3d_parametric_line

import scipy.optimize as opt
from function_quat import *

def acc_from_normal1(norm0,norm,acc,normal,center,start=[0,0,1],s_rot=np.array([0,0,0]),heuristic=False):
    X = sym.Symbol('X')
    Y = sym.Symbol('Y')
    Z = sym.Symbol('Z')
    u, v, w = symbols('u v w')
    N = sym.Matrix([X,Y,Z])#/gamma
    g = np.array([0,0,1],dtype=mpf)
    A = sym.Matrix(acc)
    OAe = A
    C = sym.Matrix(center)
    Pae = sym.simplify((OAe.dot(N))*N)
    M0 = sym.Matrix(norm0)
    M1 = sym.Matrix(norm)
    
    M0 = M0/sym.sqrt(M0.dot(M0))
    M1 = M1/sym.sqrt(M1.dot(M1))
    
    
    fe = A-Pae
    ge = skewSymmetric(N)@fe
    
    
    NM= M0
    FF = NM
    KK = v
    uFF = FF*v
    eGG = cos(v)*fe+sin(v)*ge+Pae

    teGG =eGG.subs([(X,FF[0]),(Y,FF[1]),(Z,FF[2])])
    
    duFF = uFF-sym.Matrix(start)
    duFF = duFF.dot(duFF)
    #lam_h = lambdify(v, duFF)
    #v0 = opt.minimize(lam_h, 0).x
    
    HH = sym.N(((teGG-C).dot(normal)),40)
    dHH = sym.diff(HH,v)
    ddHH = sym.diff(dHH,v)

    HHz = sym.N(((teGG-C).dot(np.array([0,0,1],dtype=mpf))),40)
    dHHz = sym.diff(HHz,v)
    
    t_v0 = start[1]
    racc= (SymExpRot2(FF,mpf(t_v0))@acc).flatten()
    cor_racc = (racc -center).dot(normal)
    pracc = racc-cor_racc*normal
    pracc = racc
    
    depth = np.array(sym.N(Pae.subs([(X,FF[0]),(Y,FF[1]),(Z,FF[2])]),40).tolist()).flatten()[1]
    
    #depth = np.array(pracc)[1]
    
    K = -center.dot(normal)
    #print("depth",depth)
    """
    P0z projection in direction z on plane
    """
    t4t0 = -1
    if normal[0]!=0 and np.abs(normal[0])>0.1:
        R = mp.norm(pracc)
        P0z = 0
        P0x = (-K-normal[1]*depth)/normal[0]
        L1 = np.array([-normal[2]/normal[0],0,1],dtype=mpf)
        P0 = np.array([P0x,depth,0],dtype=mpf)
        L1 = L1/mp.norm(L1)
        if P0x ==0:
            alpha = mp.acos(0)
        else:
            alpha =  mp.acos((sym.Matrix([-P0x,0,0]).dot(sym.Matrix(L1))/((mp.norm(P0x))*mp.norm(L1))))
        P1 = np.array(P0+P0x*mp.cos(alpha)*L1,dtype=mpf)
        
        ppracc = np.array([racc[0],0,racc[2]],dtype=mpf)
        ppracc = ppracc/mp.norm(ppracc)
        
        pnormal = np.array([normal[0],0,normal[2]],dtype=mpf)
        pnormal = pnormal/mp.norm(pnormal)
        
        gamma = mp.acos(sym.Matrix(ppracc).dot(sym.Matrix(-pnormal)))
        P1 = np.array([racc[0]*mp.cos(gamma)-racc[2]*mp.sin(gamma),depth,racc[0]*mp.sin(gamma)+racc[2]*mp.cos(gamma)],dtype=mpf)
        #P1 = P1-(np.dot(P1,normal))*normal+K
        
        e = skewSymmetric(normal)@np.array([0,1,0],dtype=mpf)
        e = e/mp.norm(e)
        direc = skewSymmetric(e)@normal
        x = (depth-center[1])/direc[1]
        P1 = x*direc+center
        
        
        if mp.norm(P1)>R:
            beta = mpf(0)
        else:
            beta = mp.acos(mp.norm(P1)/R)
        Q0 = np.array(P1+R*mp.sin(beta)*L1,dtype=mpf)
        Q1 = np.array(P1-R*mp.sin(beta)*L1,dtype=mpf)
        
        
    else:
        if normal[2] != 0:
            P0z = (-K-normal[1]*depth)/normal[2]
            L1 = np.array([1,0,-normal[0]/normal[2]],dtype=mpf)
            P0 = np.array([0,depth,P0z],dtype=mpf)
            
            L1 = L1/mp.norm(L1)
            alpha =  mp.acos((sym.Matrix([0,0,-P0z]).dot(sym.Matrix(L1))/(mp.norm(P0z)*mp.norm(L1))))
            P1 = np.array(P0+P0z*mp.cos(alpha)*L1,dtype=mpf)

            e = skewSymmetric(normal)@np.array([0,1,0],dtype=mpf)
            e = e/mp.norm(e)
            direc = skewSymmetric(e)@normal
            x = (depth-center[1])/direc[1]
            P1 = x*direc+center
            R = mp.norm(pracc)
            
            if mp.norm(P1)>R:
                beta = mpf(0)
                t4t0 = t_v0
            else:
                beta = mp.acos(mp.norm(P1)/R)
            Q0 = np.array(P1+R*mp.sin(beta)*L1,dtype=mpf)
            Q1 = np.array(P1-R*mp.sin(beta)*L1,dtype=mpf)
        else:
            Q0 = racc
            Q1 = racc
            t4t0 = t_v0
    
    ppracc = np.array([racc[0],0,racc[2]],dtype=mpf)
    ppracc = ppracc/mp.norm(ppracc)
    pQ0 = np.array([Q0[0],0,Q0[2]],dtype=mpf)
    pQ0 = pQ0/mp.norm(pQ0)
    pQ1 = np.array([Q1[0],0,Q1[2]],dtype=mpf)
    pQ1 = pQ1/mp.norm(pQ1)
    qq0 = np.array(log_q(quat_ntom(ppracc,pQ0)))
    qq1 = np.array(log_q(quat_ntom(ppracc,pQ1)))
    dang0 = mp.norm(qq0)
    dang1 = mp.norm(qq1)
    if np.linalg.norm(qq0.astype(float)) ==0:
        dang0 = 0
    else:
        v_qq0 = qq0/dang0
        dang0 = dang0*np.sign(np.dot(v_qq0,np.array(list(M0),dtype=mpf)))
    if np.linalg.norm(qq1.astype(float)) ==0:
        dang1 = 0
    else:
        v_qq1 = qq1/dang1
        dang1 = dang1*np.sign(np.dot(v_qq1,np.array(list(M0),dtype=mpf)))
    
    d_angle = dang0
    if np.abs(dang1)<np.abs(d_angle):
        d_angle = dang1
    summ = dang0+dang1
    
    
    t1t0 = mpf(d_angle+t_v0)
    t2t0 = mpf(summ-d_angle+t_v0)
    t3t0 = nsolve(dHHz,t_v0,prec=40,verify=False)
    #t5t0 = nsolve(dHHz,t_v0,prec=40,verify=False)

    t4t0=(t1t0+t2t0)/2
    t0 = t1t0
    t4t0 = t1t0
    
    rot2= (SymExpRot2(FF,t3t0))
    irot2= (SymExpRot2(FF,-t3t0))
    rot3= (SymExpRot2(FF,t2t0))
    irot3= (SymExpRot2(FF,-t2t0))
    
    
    if heuristic:
        t1t0=t3t0 #remove noise
        #sign = np.sign((center-acc).dot(normal))
        #sign = np.sign((acc).dot(normal))
        #a = ((center).dot(normal))
        #b = (teGG).dot(normal).evalf(subs={v:(t1t0+t2t0)/2})
        c = -ddHH.evalf(subs={v:t_v0})
        d = t_v0
        sign = np.sign((c))
        #print(sign,c,d)
        qq_acc = quat_ntom(np.array(normal,dtype=mpf), acc/np.linalg.norm(acc))
        FF_acc = log_q(qq_acc)
        duFF2 = uFF-sym.Matrix(-FF_acc)
        duFF2 = duFF2.dot(duFF2)
        lam_h2 = lambdify(v, duFF2)
        v0_acc = opt.minimize(lam_h2, 0).x
        
        t_v0_acc = v0_acc[0]
        #plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))),(v,-1.5,1.5),title="t3t0 "+str(t_v0))

        
        if (t1t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:t1t0})<0) or sign*(teGG-C).dot(normal).evalf(subs={v:(t1t0+t2t0)/2})<0  or np.abs(t_v0-t3t0)>np.abs(t1t0 -t2t0):
            t1t0 = t_v0
            t1t0 = t_v0_acc
            t1t0 = np.linalg.norm(FF_acc)
            FF = FF_acc/t1t0
            t1t0=-t1t0
            #rot1= (SymExpRot2(FF,-t1t0))
            #irot1= (SymExpRot2(FF,t1t0))

            
            t0 = t3t0
        #elif np.abs(t_v0_acc-t4t0)<np.abs(t_v0_acc-t3t0) and np.abs(t_v0-t4t0)<np.abs(t_v0-t3t0):
        elif np.abs(t_v0-t4t0)<np.abs(t_v0-t3t0) and np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t4t0}))<np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t3t0})) and sign*(teGG-C).dot(normal).evalf(subs={v:(t1t0+t2t0)/2})>=0:
            t1t0=t4t0
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0_acc)))),(v,-0.1,0.1),title="t_v0_acc"+str(t_v0_acc))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))),(v,-0.1,0.1),title="t_v0 "+str(t_v0))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t4t0)))),(v,-0.1,0.1),title="t1t0 "+str(t4t0))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t2t0)))),(v,-0.1,0.1),title="t2t0 "+str(t2t0))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t3t0)))),(v,-0.1,0.1),title="t3t0 "+str(t3t0))
            #rot1= (SymExpRot2(FF,t1t0))
            #irot1= (SymExpRot2(FF,-t1t0))
    rot1= (SymExpRot2(FF,t1t0))
    irot1= (SymExpRot2(FF,-t1t0))

    gamma = 0.01

    rot= (SymExpRot2(FF,t0))
    irot= (SymExpRot2(FF,-t0))
    
    return np.array(irot1,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot1,rot1,np.array(irot2,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot2,rot2,np.array(irot3,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot3,rot3


def acc_from_normal_imu(norm0,norm,acc,normal,center,start=[0,0,1],s_rot=np.array([0,0,0]),heuristic=False):
    X = sym.Symbol('X')
    Y = sym.Symbol('Y')
    Z = sym.Symbol('Z')
    u, v, w = symbols('u v w')
    N = sym.Matrix([X,Y,Z])#/gamma
    g = np.array([0,0,1],dtype=mpf)
    A = sym.Matrix(acc)
    OAe = A
    C = sym.Matrix(center)
    Pae = sym.simplify((OAe.dot(N))*N)
    M0 = sym.Matrix(norm0)
    M1 = sym.Matrix(norm)
    
    M0 = M0/sym.sqrt(M0.dot(M0))
    M1 = M1/sym.sqrt(M1.dot(M1))
    
    
    fe = A-Pae
    ge = skewSymmetric(N)@fe
    
    
    NM= M0
    FF = NM
    KK = v
    uFF = FF*v
    eGG = cos(v)*fe+sin(v)*ge+Pae

    teGG =eGG.subs([(X,FF[0]),(Y,FF[1]),(Z,FF[2])])
    
    duFF = uFF-sym.Matrix(start)
    duFF = duFF.dot(duFF)
    #lam_h = lambdify(v, duFF)
    #v0 = opt.minimize(lam_h, 0).x
    
    HH = sym.N(((teGG-C).dot(normal)),40)
    dHH = sym.diff(HH,v)
    ddHH = sym.diff(dHH,v)
    HHz = sym.N(((teGG-C).dot(np.array([0,0,1],dtype=mpf))),40)
    dHHz = sym.diff(HHz,v)
    #t_v0 = v0[0]#np.mod(v0[0],np.pi)
    t_v0 = start[1]
    racc= (SymExpRot2(FF,mpf(t_v0))@acc).flatten()
    cor_racc = (racc -center).dot(normal)
    pracc = racc-cor_racc*normal
    pracc = racc
    
    depth = np.array(sym.N(Pae.subs([(X,FF[0]),(Y,FF[1]),(Z,FF[2])]),40).tolist()).flatten()[1]
    
    
    K = -center.dot(normal)
    """
    P0z projection in direction z on plane
    """
    t4t0 = -1
    if normal[0]!=0 and np.abs(normal[0])>0.1:
        R = mp.norm(pracc)
        P0z = 0
        P0x = (-K-normal[1]*depth)/normal[0]
        L1 = np.array([-normal[2]/normal[0],0,1],dtype=mpf)
        P0 = np.array([P0x,depth,0],dtype=mpf)
        L1 = L1/mp.norm(L1)
        if P0x ==0:
            alpha = mp.acos(0)
        else:
            alpha =  mp.acos((sym.Matrix([-P0x,0,0]).dot(sym.Matrix(L1))/((mp.norm(P0x))*mp.norm(L1))))
        P1 = np.array(P0+P0x*mp.cos(alpha)*L1,dtype=mpf)
        
        ppracc = np.array([racc[0],0,racc[2]],dtype=mpf)
        ppracc = ppracc/mp.norm(ppracc)
        
        pnormal = np.array([normal[0],0,normal[2]],dtype=mpf)
        pnormal = pnormal/mp.norm(pnormal)
        
        gamma = mp.acos(sym.Matrix(ppracc).dot(sym.Matrix(-pnormal)))
        P1 = np.array([racc[0]*mp.cos(gamma)-racc[2]*mp.sin(gamma),depth,racc[0]*mp.sin(gamma)+racc[2]*mp.cos(gamma)],dtype=mpf)
        
        e = skewSymmetric(normal)@np.array([0,1,0],dtype=mpf)
        e = e/mp.norm(e)
        direc = skewSymmetric(e)@normal
        x = (depth-center[1])/direc[1]
        P1 = x*direc+center
        
        
        if mp.norm(P1)>R:
            beta = mpf(0)
        else:
            beta = mp.acos(mp.norm(P1)/R)
        Q0 = np.array(P1+R*mp.sin(beta)*L1,dtype=mpf)
        Q1 = np.array(P1-R*mp.sin(beta)*L1,dtype=mpf)
        
        
    else:
        if normal[2] != 0:
            P0z = (-K-normal[1]*depth)/normal[2]
            L1 = np.array([1,0,-normal[0]/normal[2]],dtype=mpf)
            P0 = np.array([0,depth,P0z],dtype=mpf)
            
            L1 = L1/mp.norm(L1)
            alpha =  mp.acos((sym.Matrix([0,0,-P0z]).dot(sym.Matrix(L1))/(mp.norm(P0z)*mp.norm(L1))))
            P1 = np.array(P0+P0z*mp.cos(alpha)*L1,dtype=mpf)

            
            e = skewSymmetric(normal)@np.array([0,1,0],dtype=mpf)
            e = e/mp.norm(e)
            direc = skewSymmetric(e)@normal
            x = (depth-center[1])/direc[1]
            P1 = x*direc+center
            R = mp.norm(pracc)
            
            if mp.norm(P1)>R:
                beta = mpf(0)
                t4t0 = t_v0
            else:
                beta = mp.acos(mp.norm(P1)/R)
            Q0 = np.array(P1+R*mp.sin(beta)*L1,dtype=mpf)
            Q1 = np.array(P1-R*mp.sin(beta)*L1,dtype=mpf)
        else:
            Q0 = racc
            Q1 = racc
            t4t0 = t_v0

    
    ppracc = np.array([racc[0],0,racc[2]],dtype=mpf)
    ppracc = ppracc/mp.norm(ppracc)
    pQ0 = np.array([Q0[0],0,Q0[2]],dtype=mpf)
    pQ0 = pQ0/mp.norm(pQ0)
    pQ1 = np.array([Q1[0],0,Q1[2]],dtype=mpf)
    pQ1 = pQ1/mp.norm(pQ1)
    qq0 = np.array(log_q(quat_ntom(ppracc,pQ0)))
    qq1 = np.array(log_q(quat_ntom(ppracc,pQ1)))
    dang0 = mp.norm(qq0)
    dang1 = mp.norm(qq1)
    if np.linalg.norm(qq0.astype(float)) ==0:
        dang0 = 0
    else:
        v_qq0 = qq0/dang0
        dang0 = dang0*np.sign(np.dot(v_qq0,np.array(list(M0),dtype=mpf)))
    if np.linalg.norm(qq1.astype(float)) ==0:
        dang1 = 0
    else:
        v_qq1 = qq1/dang1
        dang1 = dang1*np.sign(np.dot(v_qq1,np.array(list(M0),dtype=mpf)))
    #
    
    d_angle = dang0
    if np.abs(dang1)<np.abs(d_angle):
        d_angle = dang1
    summ = dang0+dang1
    
    
    t1t0 = mpf(d_angle+t_v0)
    t2t0 = mpf(summ-d_angle+t_v0)
    t3t0 = nsolve(dHHz,t_v0,prec=40,verify=False)
    t4t0=(t1t0+t2t0)/2
    t0 = t1t0
    t4t0 = t1t0
    
    rot2= (SymExpRot2(FF,t3t0))
    irot2= (SymExpRot2(FF,-t3t0))
    rot3= (SymExpRot2(FF,t2t0))
    irot3= (SymExpRot2(FF,-t2t0))
    
    corrected = False
    
    
    t_v0_acc = t3t0
    
    if heuristic:
        t1t0=t3t0 #remove noise
        #sign = np.sign((center-acc).dot(normal))
        #sign = np.sign((center).dot(normal))
        a = ((center).dot(normal))
        b = (teGG).dot(normal).evalf(subs={v:(t1t0+t2t0)/2})
        c = -ddHH.evalf(subs={v:t_v0})
        sign = np.sign(c)
        #print(acc,normal,center)
        #print(sign,c,b)
        qq_acc = quat_ntom(np.array([0,0,1],dtype=mpf), acc/np.linalg.norm(acc))
        FF_acc = log_q(qq_acc)
        duFF2 = uFF-sym.Matrix(-FF_acc)
        duFF2 = duFF2.dot(duFF2)
        lam_h2 = lambdify(v, duFF2)
        v0_acc = opt.minimize(lam_h2, 0).x
        t_v0_acc = v0_acc[0]
        t_v0_acc = -FF_acc[1]
        print("ff acc",-FF_acc)
        gamma = 1.0
        #plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))),(v,-0.1,0.1),title="t_v0 "+str(t_v0))
        print("angles",t_v0,t3t0,t4t0,t_v0_acc)
        print("t1t0",t1t0,t2t0)
        t1t0 = t_v0_acc
        prob = np.random.random(1)
        prob = 0
        print("prob",prob,sign*(teGG-C).dot(normal).evalf(subs={v:(t1t0+t2t0)/2})<0,np.abs(t_v0-t3t0)>np.abs(t1t0 -t2t0)*1.5,(t1t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:t1t0})<0))
        print("norms",(np.abs((teGG).dot(normal).evalf(subs={v:t_v0}))),(np.abs((C).dot(normal))))
        if (t1t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:t1t0})<0) or sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0  or np.abs(t_v0-t3t0)>np.abs(t1t0 -t2t0)*1.5:
            print("case1",np.abs(t_v0-t3t0),np.abs(t1t0 -t2t0),sign*(teGG-C).dot(normal).evalf(subs={v:t_v0})<0)
            t1t0 = t_v0
            t1t0 = t_v0_acc
            t1t0 = np.linalg.norm(FF_acc)
            FF = FF_acc/t1t0
            t1t0=-t1t0
            print("t1t0",t1t0,t_v0)
            #rot1= (SymExpRot2(FF,-t1t0))
            #irot1= (SymExpRot2(FF,t1t0))

            
            t0 = t3t0
        elif (np.abs(t_v0-t4t0)<np.abs(t_v0-t3t0)*gamma and np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t4t0}))<np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t3t0}))*gamma and sign*(teGG-C).dot(normal).evalf(subs={v:t_v0})>=0):
            print("case 2")
            t1t0=t4t0
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t1t0)))),(v,-0.1,0.1),title="t1t0 "+str(t1t0))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t3t0)))),(v,-0.1,0.1),title="t3t0 "+str(t3t0))
            """plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0_acc)))),(v,-0.1,0.1),title="t_v0_acc"+str(t_v0_acc))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))),(v,-0.1,0.1),title="t_v0 "+str(t_v0))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t4t0)))),(v,-0.1,0.1),title="t1t0 "+str(t4t0))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t2t0)))),(v,-0.1,0.1),title="t2t0 "+str(t2t0))
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t3t0)))),(v,-0.1,0.1),title="t3t0 "+str(t3t0))"""
            corrected=True 
        elif prob>0.95:
            t1t0=t4t0
                
    rot1= (SymExpRot2(FF,t1t0))
    irot1= (SymExpRot2(FF,-t1t0))
    

    
    gamma = 0.01
    rot= (SymExpRot2(FF,t0))
    irot= (SymExpRot2(FF,-t0))
    
    return np.array(irot1,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot1,rot1,np.array(irot2,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot2,rot2,np.array(irot3,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot3,rot3,corrected,np.abs(t_v0_acc-t1t0)
