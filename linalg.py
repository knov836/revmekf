
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
import pdb


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
    HHt = sym.N(((teGG-C).dot(normal))**2,40)
    dHHt = sym.diff(HHt,v)
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
            #R = mp.norm([pracc[0],pracc[2]])
            R = mp.norm(pracc)
            
            
            if mp.norm(P1)>R:
                beta = mpf(0)
                t4t0 = t_v0
            else:
                beta = mp.acos(mp.norm(P1)/R)
                #beta = mp.acos(mp.norm([P1[0],P1[2]])/R)
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
    t5t0 = nsolve(dHHt,t_v0,prec=40,verify=False)
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
        t1t0 = t_v0_acc
        #print("ff acc",-FF_acc)
        gamma = 1.0
        
        #plot(HH,(v,-1.5,1.5))
        l_HH = (teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))
        x_pts = [float(t1t0 - t_v0), float(t4t0 - t_v0), float(t2t0 - t_v0),float(t5t0 - t_v0)]
        y_pts = np.array([l_HH.evalf(subs={v: float(t1t0 - t_v0)}),
         l_HH.evalf(subs={v: float(t4t0 - t_v0)}),
         l_HH.evalf(subs={v: float(t2t0 - t_v0)}),
         l_HH.evalf(subs={v: float(t5t0 - t_v0)})]).astype(float)
        
        labels = ["t1", "t4", "t2","t5"]
        f = sym.lambdify(v, l_HH, 'numpy')
        vv = np.linspace(-0.5, 0.5, 200)
        
        
        prob = np.random.random(1)
        prob = 0
        
        #print("prob",prob,sign*(teGG-C).dot(normal).evalf(subs={v:(t1t0+t2t0)/2})<0,np.abs(t_v0-t3t0)>np.abs(t1t0 -t2t0)*1.5,(t1t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:t1t0})<0))
        #print("norms",(np.abs((teGG).dot(normal).evalf(subs={v:t_v0}))),(np.abs((C).dot(normal))))
        print("racc",np.abs(np.linalg.norm(np.array([racc[0],racc[2]]).astype(float))/np.linalg.norm(center.astype(float)))>20,np.linalg.norm(np.array([racc[0],racc[2]]).astype(float)), np.linalg.norm(center.astype(float)))
        if (t4t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0) or sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0  or np.abs(t_v0-t3t0)>np.abs(t4t0 -t2t0)*1.5:
            #print("case1",np.abs(t_v0-t3t0),np.abs(t1t0 -t2t0),sign*(teGG-C).dot(normal).evalf(subs={v:t_v0})<0)
            t1t0 = t_v0
            t1t0 = t_v0_acc
            t1t0 = np.linalg.norm(FF_acc)
            FF = FF_acc/t1t0
            t1t0=-t1t0

        elif (((np.abs(t_v0-t4t0)<np.abs(t_v0-t1t0)*gamma and np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t4t0}))<np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t1t0}))*gamma)) and sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})>=0):
            t1t0=t4t0
            #plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t1t0)))),(v,-0.1,0.1),title="t1t0 "+str(t1t0))
            #plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t3t0)))),(v,-0.1,0.1),title="t3t0 "+str(t3t0))
            corrected=True 
        if np.abs(t4t0-t_v0)>np.pi/4:
            t1t0 = t4t0
            corrected = True
            
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(vv, f(vv), label='l_HH(v)', color='blue')
            ax.scatter(x_pts, y_pts, color='red', marker='o', s=60,label='interesting points')
            for x, y, label in zip(x_pts, y_pts, labels):
                ax.annotate(label,
                    xy=(x, y),                # position du point
                    xytext=(0, 8),            # décalage texte en points
                    textcoords='offset points',
                    ha='center',
                    color='green',
                    fontsize=12)
            # Axes, grille, labels
            ax.axhline(0, color='black', linewidth=0.8)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel("v")
            ax.set_ylabel("l_HH(v)")
            ax.set_title("l_HH")
            ax.legend()
        """else:
            t1t0 = t_v0_acc
            t1t0 = np.linalg.norm(FF_acc)
            FF = FF_acc/t1t0
            t1t0=-t1t0"""
    """if np.abs(t1t0-t_v0)>np.pi/4:
        
        pdb.set_trace()"""
    rot1= (SymExpRot2(FF,t1t0))
    irot1= (SymExpRot2(FF,-t1t0))
    

    
    rot= (SymExpRot2(FF,t0))
    irot= (SymExpRot2(FF,-t0))
    
    return np.array(irot1,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot1,rot1,np.array(irot2,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot2,rot2,np.array(irot3,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot3,rot3,corrected,np.abs(t_v0_acc-t1t0)

def acc_from_normal_imu_grav_neural(norm0,norm,acc,grav,normal,center,start=[0,0,1],s_rot=np.array([0,0,0]),heuristic=False,correction=False):
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
    HHt = sym.N(((teGG-C).dot(normal))**2,40)
    dHHt = sym.diff(HHt,v)
    dHH = sym.diff(HH,v)
    ddHH = sym.diff(dHH,v)
    HHz = sym.N(((teGG-C).dot(np.array([0,0,1],dtype=mpf))),40)
    dHHz = sym.diff(HHz,v)
    #t_v0 = v0[0]#np.mod(v0[0],np.pi)
    t_v0 = start[1]
    racc= (SymExpRot2(FF,mpf(t_v0))@acc).flatten()
    rgrav= (SymExpRot2(FF,mpf(t_v0))@grav).flatten()
    cor_racc = (racc -center).dot(normal)
    #pracc = racc-cor_racc*normal
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
            #alpha =  mp.acos((sym.Matrix([0,0,-P0z]).dot(sym.Matrix(L1))/(mp.norm(P0z)*mp.norm(L1))))
            #P1 = np.array(P0+P0z*mp.cos(alpha)*L1,dtype=mpf)

            
            e = skewSymmetric(normal)@np.array([0,1,0],dtype=mpf)
            e = e/mp.norm(e)
            direc = skewSymmetric(e)@normal
            x = (depth-center[1])/direc[1]
            P1 = x*direc+center
            #R = mp.norm([pracc[0],pracc[2]])
            R = mp.norm(pracc)
            Rg = mp.norm(rgrav)
            
            if mp.norm(P1)>R:
                beta = mpf(0)
                t4t0 = t_v0
            else:
                beta = mp.acos(mp.norm(P1)/R)
                #beta = mp.acos(mp.norm([P1[0],P1[2]])/R)
            Q0 = np.array(P1+R*mp.sin(beta)*L1,dtype=mpf)
            Q1 = np.array(P1-R*mp.sin(beta)*L1,dtype=mpf)
            
            
            if mp.norm(P1)>Rg:
                betag = mpf(0)
                t4t0g = t_v0
            else:
                betag = mp.acos(mp.norm(P1)/Rg)
                #beta = mp.acos(mp.norm([P1[0],P1[2]])/R)
            Qg0 = np.array(P1+Rg*mp.sin(betag)*L1,dtype=mpf)
            Qg1 = np.array(P1-Rg*mp.sin(betag)*L1,dtype=mpf)
        else:
            Q0 = racc
            Q1 = racc
            t4t0 = t_v0
            Qg0 = rgrav
            Qg1 = rgrav
            t4t0g = t_v0

    
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
    
    ppracc = np.array([rgrav[0],0,rgrav[2]],dtype=mpf)
    ppracc = ppracc/mp.norm(ppracc)
    pQ0 = np.array([Qg0[0],0,Qg0[2]],dtype=mpf)
    pQ0 = pQ0/mp.norm(pQ0)
    pQ1 = np.array([Qg1[0],0,Qg1[2]],dtype=mpf)
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
    
    
    t1t0g = mpf(d_angle+t_v0)
    t2t0g = mpf(summ-d_angle+t_v0)
    
    
    
    t3t0 = nsolve(dHHz,t_v0,prec=40,verify=False)
    t5t0 = nsolve(dHHt,t_v0,prec=40,verify=False)
    t0 = t1t0
    t4t0 = t1t0
    
    rot2= (SymExpRot2(FF,t3t0))
    irot2= (SymExpRot2(FF,-t3t0))
    rot3= (SymExpRot2(FF,t2t0))
    irot3= (SymExpRot2(FF,-t2t0))
    
    corrected = False
    not_corrected = False
    
    t_v0_acc = t3t0
    
    rt0 =0
    rt2 =0
    rt3 =0
    rt4 =0
    ert0 =0
    ert2 =0
    ert3 =0
    ert4 =0
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
        t1t0 = t_v0_acc
        
        """t1t0 = np.linalg.norm(FF_acc)
        FF = FF_acc/t1t0
        t1t0=-t1t0"""
        
        
        
        #print("ff acc",-FF_acc)
        gamma = 1.0
        
        #plot(HH,(v,-1.5,1.5))
        

        #print("angles",t_v0,t3t0,t4t0,t_v0_acc)
        #print("t1t0",t1t0,t2t0)
        
        prob = np.random.random(1)
        prob = 0
        rt0 = t_v0
        ert0 = HH.evalf(subs={v: float(t_v0)})
        
        rt2 = t2t0-t_v0
        ert0 = HH.evalf(subs={v: float(t2t0)})
        
        rt3 = t1t0-t_v0
        ert0 = HH.evalf(subs={v: float(t1t0)})
        
        rt4 = t4t0-t_v0
        ert0 = HH.evalf(subs={v: float(t4t0)})
        #print("prob",prob,sign*(teGG-C).dot(normal).evalf(subs={v:(t1t0+t2t0)/2})<0,np.abs(t_v0-t3t0)>np.abs(t1t0 -t2t0)*1.5,(t1t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:t1t0})<0))
        #print("norms",(np.abs((teGG).dot(normal).evalf(subs={v:t_v0}))),(np.abs((C).dot(normal))))
        #print("racc",np.abs(np.linalg.norm(np.array([racc[0],racc[2]]).astype(float))/np.linalg.norm(center.astype(float)))>20,np.linalg.norm(np.array([racc[0],racc[2]]).astype(float)), np.linalg.norm(center.astype(float)))
        
        if correction:
            t1t0 = np.sign(float(t4t0-t_v0))*np.abs(float(np.abs(t4t0-t_v0)))/2+t_v0
            corrected=True
            l_HH = (teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))
            x_pts = [float(t_v0_acc - t_v0),float(t4t0 - t_v0), float(t2t0 - t_v0)]
            y_pts = np.array([l_HH.evalf(subs={v: float(t_v0_acc - t_v0)}),
             l_HH.evalf(subs={v: float(t4t0 - t_v0)}),
             l_HH.evalf(subs={v: float(t2t0 - t_v0)})]).astype(float)
            
            labels = ["t_v0_acc", "t4", "t2"]
            tt = np.max([np.abs(float(t4t0)),np.abs(float(t2t0))])
            f = sym.lambdify(v, l_HH, 'numpy')
            vv = np.linspace(-tt, tt, 1000)
            if correction:
                t1t0 = np.sign(float(t4t0-t_v0))*np.abs(float(np.abs(t4t0-t_v0)))/2+t_v0
                corrected=True 

                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(vv, f(vv), label='l_HH(v)', color='blue')
                ax.scatter(x_pts, y_pts, color='red', marker='o', s=60,label='interesting points')
                
                for x, y, label in zip(x_pts, y_pts, labels):
                    ax.annotate(label,
                        xy=(x, y),                # position du point
                        xytext=(0, 8),            # décalage texte en points
                        textcoords='offset points',
                        ha='center',
                        color='green',
                        fontsize=12)
                # Axes, grille, labels
                ax.axhline(0, color='black', linewidth=0.8)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_xlabel("v")
                ax.set_ylabel("l_HH(v)")
                ax.set_title("l_HH")
                ax.legend()
        else:
            t1t0 = t_v0
            t1t0 = t_v0_acc
            t1t0 = np.linalg.norm(FF_acc)
            FF = FF_acc/t1t0
            t1t0=-t1t0
            

    rot1= (SymExpRot2(FF,t1t0))
    irot1= (SymExpRot2(FF,-t1t0))
    

    
    rot= (SymExpRot2(FF,t0))
    irot= (SymExpRot2(FF,-t0))
    
    return np.array(irot1,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot1,rot1,np.array(irot2,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot2,rot2,np.array(irot3,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot3,rot3,corrected,not_corrected,np.abs(t_v0_acc-t1t0),rt0,rt2,rt3,rt4,ert0,ert2,ert3,ert4


def acc_from_normal_imu_grav(norm0,norm,acc,grav,normal,center,start=[0,0,1],s_rot=np.array([0,0,0]),heuristic=False,correction=False):
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
    HHt = sym.N(((teGG-C).dot(normal))**2,40)
    dHHt = sym.diff(HHt,v)
    dHH = sym.diff(HH,v)
    ddHH = sym.diff(dHH,v)
    HHz = sym.N(((teGG-C).dot(np.array([0,0,1],dtype=mpf))),40)
    dHHz = sym.diff(HHz,v)
    #t_v0 = v0[0]#np.mod(v0[0],np.pi)
    t_v0 = start[1]
    racc= (SymExpRot2(FF,mpf(t_v0))@acc).flatten()
    rgrav= (SymExpRot2(FF,mpf(t_v0))@grav).flatten()
    cor_racc = (racc -center).dot(normal)
    #pracc = racc-cor_racc*normal
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
            #alpha =  mp.acos((sym.Matrix([0,0,-P0z]).dot(sym.Matrix(L1))/(mp.norm(P0z)*mp.norm(L1))))
            #P1 = np.array(P0+P0z*mp.cos(alpha)*L1,dtype=mpf)

            
            e = skewSymmetric(normal)@np.array([0,1,0],dtype=mpf)
            e = e/mp.norm(e)
            direc = skewSymmetric(e)@normal
            x = (depth-center[1])/direc[1]
            P1 = x*direc+center
            #R = mp.norm([pracc[0],pracc[2]])
            R = mp.norm(pracc)
            Rg = mp.norm(rgrav)
            
            if mp.norm(P1)>R:
                beta = mpf(0)
                t4t0 = t_v0
            else:
                beta = mp.acos(mp.norm(P1)/R)
                #beta = mp.acos(mp.norm([P1[0],P1[2]])/R)
            Q0 = np.array(P1+R*mp.sin(beta)*L1,dtype=mpf)
            Q1 = np.array(P1-R*mp.sin(beta)*L1,dtype=mpf)
            
            
            if mp.norm(P1)>Rg:
                betag = mpf(0)
                t4t0g = t_v0
            else:
                betag = mp.acos(mp.norm(P1)/Rg)
                #beta = mp.acos(mp.norm([P1[0],P1[2]])/R)
            Qg0 = np.array(P1+Rg*mp.sin(betag)*L1,dtype=mpf)
            Qg1 = np.array(P1-Rg*mp.sin(betag)*L1,dtype=mpf)
        else:
            Q0 = racc
            Q1 = racc
            t4t0 = t_v0
            Qg0 = rgrav
            Qg1 = rgrav
            t4t0g = t_v0

    
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
    
    ppracc = np.array([rgrav[0],0,rgrav[2]],dtype=mpf)
    ppracc = ppracc/mp.norm(ppracc)
    pQ0 = np.array([Qg0[0],0,Qg0[2]],dtype=mpf)
    pQ0 = pQ0/mp.norm(pQ0)
    pQ1 = np.array([Qg1[0],0,Qg1[2]],dtype=mpf)
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
    
    
    t1t0g = mpf(d_angle+t_v0)
    t2t0g = mpf(summ-d_angle+t_v0)
    
    
    
    t3t0 = nsolve(dHHz,t_v0,prec=40,verify=False)
    t5t0 = nsolve(dHHt,t_v0,prec=40,verify=False)
    t0 = t1t0
    t4t0 = t1t0
    
    rot2= (SymExpRot2(FF,t3t0))
    irot2= (SymExpRot2(FF,-t3t0))
    rot3= (SymExpRot2(FF,t2t0))
    irot3= (SymExpRot2(FF,-t2t0))
    
    corrected = False
    not_corrected = False
    
    t_v0_acc = t3t0
    rt0 =0
    rt2 =0
    rt3 =0
    rt4 =0
    ert0 =0
    ert2 =0
    ert3 =0
    ert4 =0
    
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
        t1t0 = t_v0_acc
        
        """t1t0 = np.linalg.norm(FF_acc)
        FF = FF_acc/t1t0
        t1t0=-t1t0"""
        
        
        gamma = 0.5
        
        prob = np.random.random(1)
        prob = 0
        l_HH = (teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))
        rt0 = t_v0
        ert0 = HH.evalf(subs={v: float(t_v0)})
        
        rt2 = t2t0-t_v0
        ert0 = HH.evalf(subs={v: float(t2t0)})
        
        rt3 = t1t0-t_v0
        ert0 = HH.evalf(subs={v: float(t1t0)})
        
        rt4 = t4t0-t_v0
        ert0 = HH.evalf(subs={v: float(t4t0)})
        
        
        x_pts = [float(t_v0_acc - t_v0),float(t4t0 - t_v0), float(t2t0 - t_v0)]
        y_pts = np.array([l_HH.evalf(subs={v: float(t_v0_acc - t_v0)}),
         l_HH.evalf(subs={v: float(t4t0 - t_v0)}),
         l_HH.evalf(subs={v: float(t2t0 - t_v0)})]).astype(float)
        
        labels = ["MEKF", "Theta 1", "Theta 2"]
        tt = np.max([np.abs(float(t4t0)),np.abs(float(t2t0))])
        f = sym.lambdify(v, l_HH, 'numpy')
        vv = np.linspace(-tt, tt, 1000)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(vv, f(vv), label='h_k(Theta)', color='blue')
        ax.scatter(x_pts, y_pts, color='red', marker='o', s=60,label='Intersection points')
        
        for x, y, label in zip(x_pts, y_pts, labels):
            ax.annotate(label,
                xy=(x, y),                # position du point
                xytext=(0, 8),            # décalage texte en points
                textcoords='offset points',
                ha='center',
                color='green',
                fontsize=12)
        # Axes, grille, labels
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("Theta")
        ax.set_ylabel("meters")
        ax.set_title("h_k")
        ax.legend()
        plt.show()
        print("angles",t4t0,t_v0_acc,t0)
            
        if (t4t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0) or sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0  or np.abs(t_v0-t3t0)>np.abs(t4t0 -t2t0)*1.5:
            t1t0 = t_v0
            t1t0 = t_v0_acc
            t1t0 = np.linalg.norm(FF_acc)
            FF = FF_acc/t1t0
            t1t0=-t1t0
            
            t0 = t3t0
        #elif (((np.abs(t_v0-t4t0)<np.abs(t_v0-t1t0)*gamma and np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t4t0}))<np.abs((teGG-C).dot(normal).evalf(subs={v:t_v0})-(teGG-C).dot(normal).evalf(subs={v:t1t0})))) and sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})>=0):
        elif (np.abs(t_v0-t4t0)<np.abs(t_v0-t1t0)*gamma and sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})>=0):
            t1t0=t4t0
            plot(((teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))),(v,-0.1,0.1),title="t_v0 "+str(t_v0))

            corrected=True 
            
        else:
            if not((t4t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0) or sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0 ):
                
                    
                
                x_pts = [float(t_v0_acc - t_v0),float(t4t0 - t_v0), float(t2t0 - t_v0)]
                y_pts = np.array([l_HH.evalf(subs={v: float(t_v0_acc - t_v0)}),
                 l_HH.evalf(subs={v: float(t4t0 - t_v0)}),
                 l_HH.evalf(subs={v: float(t2t0 - t_v0)})]).astype(float)
                
                labels = ["t_v0_acc", "t4", "t2"]
                tt = np.max([np.abs(float(t4t0)),np.abs(float(t2t0))])
                f = sym.lambdify(v, l_HH, 'numpy')
                vv = np.linspace(-tt, tt, 1000)
                if (np.abs(t4t0-t_v0)>np.pi/10 and np.abs(t4t0-t_v0)<np.pi/2 and np.abs(t_v0-(t4t0+t2t0)/2)>np.pi/20) or (np.abs(t2t0-t_v0)>2*np.abs(t4t0-t_v0) and 3*np.abs(t_v0_acc-t_v0)>np.abs(t4t0-t_v0) and np.abs(t4t0-t2t0)>0.4):
                    print(np.abs(t_v0_acc-t_v0),np.abs(t4t0-t_v0))
                    #t1t0 = np.sign(float(t4t0-t_v0))*np.log(float(1+np.abs(t4t0-t_v0)))+t_v0
                    #t1t0 = np.sign(float(t4t0-t_v0))*np.abs(float(np.abs(t4t0-t_v0)))/2+t_v0
                    #corrected = True
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.plot(vv, f(vv), label='l_HH(v)', color='blue')
                    ax.scatter(x_pts, y_pts, color='red', marker='o', s=60,label='interesting points')
                    
                    for x, y, label in zip(x_pts, y_pts, labels):
                        ax.annotate(label,
                            xy=(x, y),                # position du point
                            xytext=(0, 8),            # décalage texte en points
                            textcoords='offset points',
                            ha='center',
                            color='green',
                            fontsize=12)
                    # Axes, grille, labels
                    ax.axhline(0, color='black', linewidth=0.8)
                    ax.axvline(0, color='black', linewidth=0.8)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.set_xlabel("v")
                    ax.set_ylabel("l_HH(v)")
                    ax.set_title("l_HH")
                    ax.legend()
                #pdb.set_trace()

    rot1= (SymExpRot2(FF,t1t0))
    irot1= (SymExpRot2(FF,-t1t0))
    

    
    rot= (SymExpRot2(FF,t0))
    irot= (SymExpRot2(FF,-t0))
    
    return np.array(irot1,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot1,rot1,np.array(irot2,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot2,rot2,np.array(irot3,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot3,rot3,corrected,not_corrected,np.abs(t_v0_acc-t1t0),rt0,rt2,rt3,rt4,ert0,ert2,ert3,ert4



def acc_from_normal_imu_grav_manual(norm0,norm,acc,grav,normal,center,start=[0,0,1],s_rot=np.array([0,0,0]),heuristic=False,correction=False):
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
    HHt = sym.N(((teGG-C).dot(normal))**2,40)
    dHHt = sym.diff(HHt,v)
    dHH = sym.diff(HH,v)
    ddHH = sym.diff(dHH,v)
    HHz = sym.N(((teGG-C).dot(np.array([0,0,1],dtype=mpf))),40)
    dHHz = sym.diff(HHz,v)
    #t_v0 = v0[0]#np.mod(v0[0],np.pi)
    t_v0 = start[1]
    racc= (SymExpRot2(FF,mpf(t_v0))@acc).flatten()
    rgrav= (SymExpRot2(FF,mpf(t_v0))@grav).flatten()
    cor_racc = (racc -center).dot(normal)
    #pracc = racc-cor_racc*normal
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
            #alpha =  mp.acos((sym.Matrix([0,0,-P0z]).dot(sym.Matrix(L1))/(mp.norm(P0z)*mp.norm(L1))))
            #P1 = np.array(P0+P0z*mp.cos(alpha)*L1,dtype=mpf)

            
            e = skewSymmetric(normal)@np.array([0,1,0],dtype=mpf)
            e = e/mp.norm(e)
            direc = skewSymmetric(e)@normal
            x = (depth-center[1])/direc[1]
            P1 = x*direc+center
            #R = mp.norm([pracc[0],pracc[2]])
            R = mp.norm(pracc)
            Rg = mp.norm(rgrav)
            
            if mp.norm(P1)>R:
                beta = mpf(0)
                t4t0 = t_v0
            else:
                beta = mp.acos(mp.norm(P1)/R)
                #beta = mp.acos(mp.norm([P1[0],P1[2]])/R)
            Q0 = np.array(P1+R*mp.sin(beta)*L1,dtype=mpf)
            Q1 = np.array(P1-R*mp.sin(beta)*L1,dtype=mpf)
            
            
            if mp.norm(P1)>Rg:
                betag = mpf(0)
                t4t0g = t_v0
            else:
                betag = mp.acos(mp.norm(P1)/Rg)
                #beta = mp.acos(mp.norm([P1[0],P1[2]])/R)
            Qg0 = np.array(P1+Rg*mp.sin(betag)*L1,dtype=mpf)
            Qg1 = np.array(P1-Rg*mp.sin(betag)*L1,dtype=mpf)
        else:
            Q0 = racc
            Q1 = racc
            t4t0 = t_v0
            Qg0 = rgrav
            Qg1 = rgrav
            t4t0g = t_v0

    
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
    
    ppracc = np.array([rgrav[0],0,rgrav[2]],dtype=mpf)
    ppracc = ppracc/mp.norm(ppracc)
    pQ0 = np.array([Qg0[0],0,Qg0[2]],dtype=mpf)
    pQ0 = pQ0/mp.norm(pQ0)
    pQ1 = np.array([Qg1[0],0,Qg1[2]],dtype=mpf)
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
    
    
    t1t0g = mpf(d_angle+t_v0)
    t2t0g = mpf(summ-d_angle+t_v0)
    
    
    
    t3t0 = nsolve(dHHz,t_v0,prec=40,verify=False)
    t5t0 = nsolve(dHHt,t_v0,prec=40,verify=False)
    t0 = t1t0
    t4t0 = t1t0
    
    rot2= (SymExpRot2(FF,t3t0))
    irot2= (SymExpRot2(FF,-t3t0))
    rot3= (SymExpRot2(FF,t2t0))
    irot3= (SymExpRot2(FF,-t2t0))
    
    corrected = False
    not_corrected = False
    label=""
    
    t_v0_acc = t3t0
    rt0 =0
    rt2 =0
    rt3 =0
    rt4 =0
    ert0 =0
    ert2 =0
    ert3 =0
    ert4 =0
    
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
        t1t0 = t_v0_acc
        
        """t1t0 = np.linalg.norm(FF_acc)
        FF = FF_acc/t1t0
        t1t0=-t1t0"""
        
        
        gamma = 1.0
        
        prob = np.random.random(1)
        prob = 0
        l_HH = (teGG-C).dot(normal).subs(v,v+sym.Rational(float(t_v0)))
        rt0 = t_v0
        ert0 = HH.evalf(subs={v: float(t_v0)})
        
        rt2 = t2t0-t_v0
        ert0 = HH.evalf(subs={v: float(t2t0)})
        
        rt3 = t1t0-t_v0
        ert0 = HH.evalf(subs={v: float(t1t0)})
        
        rt4 = t4t0-t_v0
        ert0 = HH.evalf(subs={v: float(t4t0)})
        
        
        x_pts = [float(t_v0_acc - t_v0),float(t4t0 - t_v0), float(t2t0 - t_v0)]
        y_pts = np.array([l_HH.evalf(subs={v: float(t_v0_acc - t_v0)}),
         l_HH.evalf(subs={v: float(t4t0 - t_v0)}),
         l_HH.evalf(subs={v: float(t2t0 - t_v0)})]).astype(float)
        
        labels = ["t_v0_acc", "t4", "t2"]
        tt = np.max([np.abs(float(t4t0)),np.abs(float(t2t0))])
        f = sym.lambdify(v, l_HH, 'numpy')
        vv = np.linspace(-tt, tt, 1000)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(vv, f(vv), label='l_HH(v)', color='blue')
        ax.scatter(x_pts, y_pts, color='red', marker='o', s=60,label='interesting points')
        
        for x, y, lab in zip(x_pts, y_pts, labels):
            ax.annotate(lab,
                xy=(x, y),                # position du point
                xytext=(0, 8),            # décalage texte en points
                textcoords='offset points',
                ha='center',
                color='green',
                fontsize=12)
        # Axes, grille, labels
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("v")
        ax.set_ylabel("l_HH(v)")
        ax.set_title("l_HH")
        ax.legend()
        plt.show()
        print("norm",mp.norm(racc)*10000)
        print("angles",rt0,rt2,rt3,rt4,ert0,ert2,ert3,ert4)
        pdb.set_trace()
        """
        corrected,not_corrected,label= False,False,""
        corrected,not_corrected,label= True,False,""
        corrected,not_corrected,label= False,True,""
        """
        
        if (t4t0 == t2t0 and sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0) or sign*(teGG-C).dot(normal).evalf(subs={v:(t4t0+t2t0)/2})<0  or np.abs(t_v0-t3t0)>np.abs(t4t0 -t2t0)*1.5:
            t1t0 = t_v0
            t1t0 = t_v0_acc
            t1t0 = np.linalg.norm(FF_acc)
            FF = FF_acc/t1t0
            t1t0=-t1t0
            #not_corrected = True
        else:
            if (corrected):
                print(np.abs(t_v0_acc-t_v0),np.abs(t4t0-t_v0))
                t1t0 = np.sign(float(t4t0-t_v0))*np.abs(float(np.abs(t4t0-t_v0)))/2+t_v0
        

    rot1= (SymExpRot2(FF,t1t0))
    irot1= (SymExpRot2(FF,-t1t0))
    

    
    rot= (SymExpRot2(FF,t0))
    irot= (SymExpRot2(FF,-t0))
    
    return np.array(irot1,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot1,rot1,np.array(irot2,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot2,rot2,np.array(irot3,dtype=mpf)@np.array([0,0,1],dtype=mpf),irot3,rot3,corrected,not_corrected,label,np.abs(t_v0_acc-t1t0),rt0,rt2,rt3,rt4,ert0,ert2,ert3,ert4
