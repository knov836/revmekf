import numpy as np
from mpmath import mp
from mpmath import mpf
from function_quat import *
import sympy as sp
import pdb
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])
def compute_normals(N,Acc,gravity,Mag,mag0):
    normals = np.zeros((N,3))

    for i in range(0,N,1):
        ss = 500
        if i<N-1-ss:
            #acc_mean = np.mean(df[i:ss+i,1:4],axis=0)
            acc_mean = np.mean(Acc[i:ss+i,0:3],axis=0)
            a = np.copy(acc_mean)
            a=a/np.linalg.norm(a)
            mag_mean = np.mean(Mag[i:ss+i,0:3],axis=0)
            h_mag = mag_mean[:2]
            h_axis = np.cross(h_mag,zaxis)
            h_axis = h_axis/np.linalg.norm(h_axis)
            
            mag_mean = mag_mean /np.linalg.norm(mag_mean)
            #print(mag_mean)
            mag_mean = np.array(quat_rot([0,*mag_mean],ExpQua(h_axis*np.arcsin(mag0[2]))))[1:4]
            #print(mag_mean,np.array(quat_rot([0,*mag0],ExpQua(h_axis*np.arcsin(mag0[2]))))[1:4])
            #pdb.set_trace()
            #break
            acc_mean = acc_mean /np.linalg.norm(gravity)
            alpha = np.sqrt(np.max(np.abs(1-acc_mean[2]**2),0))
            
            angle_alpha= np.arccos(alpha)
            
            """angle_beta = np.pi/2-angle_alpha
            beta = np.cos(angle_beta)
            angle_gamma = np.arccos(np.sign(mag_mean[2])*np.max(np.abs(mag_mean[2]),np.abs(beta))/beta)"""
            
            
            qq1 = quat_ntom( np.array([1,0,0]),mag_mean)
            rotated_z = np.array(quat_rot(np.array([0,0,0,1]),(qq1)))[1:4]
            
            
            axis1 = mag_mean
            
            pacc = np.dot(rotated_z,axis1)*axis1
            oacc = rotated_z-pacc
            #normalized_oacc = oacc /np.linalg.norm(oacc)
            
            paz = rotated_z[2] - pacc[2]
            
            n1 =axis1
            n2 = zaxis
            d1 = -np.dot(rotated_z,axis1)
            d2 = -acc_mean[2]
            point, direction = intersection_line_from_planes(n1, d1, n2, d2)
            
            oacc= oacc/np.linalg.norm(oacc)
            direction = direction/np.linalg.norm(direction)
            t = sp.Symbol('t', real=True)
            P = sp.Matrix(point)
            d = sp.Matrix(direction)
            A = sp.Matrix(pacc)
            
            X_t = P + t * d
            R = np.linalg.norm(oacc)
            eq = sp.N((X_t - A).dot(X_t - A)-R**2,40)
            sol0 = sp.nsolve(sp.diff(eq),0)
            sol1 = -sol0
            
            solutions=[sol0,sol1]
            points = [np.array(X_t.subs(t, sol)).flatten() for sol in solutions]
            
            tthetas= np.zeros(len(points))
            for k in range(len(points)):
                p = points[k]
                dd = p-pacc
                dd = dd/np.linalg.norm(dd.astype(float))*np.linalg.norm(oacc.astype(float))
                theta0 = np.arccos(np.sign(np.dot(oacc.astype(float),dd.astype(float)))*np.min([np.abs(np.dot(oacc.astype(float),dd.astype(float))),1]))
                theta1 = -theta0
                v0 = np.array(quat_rot([0,*oacc],ExpQua(theta0*axis1)))[1:4]
                v1 = np.array(quat_rot([0,*oacc],ExpQua(theta1*axis1)))[1:4]
                if (np.abs(np.dot(v0,np.array(dd).astype(float))))<(np.abs(np.dot(v1,np.array(dd).astype(float)))):
                    tthetas[k] = theta1
                else:
                    tthetas[k] = theta0

            ttheta = 0
            theta0 = tthetas[0]
            theta1 = tthetas[1]
            v0 = np.array(quat_rot([0,*oacc],ExpQua(theta0*axis1)))[1:4]
            v1 = np.array(quat_rot([0,*oacc],ExpQua(theta1*axis1)))[1:4]
            if (np.abs(np.dot(v0,np.array(a).astype(float))))<(np.abs(np.dot(v1,np.array(a).astype(float)))):
                ttheta = theta1
            else:
                ttheta = theta0
                
            qq2 = quat_mult(ExpQua(ttheta*axis1), qq1)
            normal = np.array(quat_rot([0,0,0,1], quat_inv(qq2)))[1:4]
            normals[i,:] = normal/np.linalg.norm(normal)

        else:
            normals[i,:]= normals[i-1,:]
    return normals

N0=1000
compute_normals(N0, np.random.rand(N0,3), np.repeat(1,3),np.random.rand(N0,3),np.array([1,0,0]))