#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def create_single_system(tr0, P):
    # Create the system Ax = b
    # Append on python list is faster than on np.array
    # Conversion is done after the loops
    A = []
    b = []
 
    # trajectory equations
    for i in range(len(tr0[0])):
        x = tr0[0][i]
        y = abs(tr0[1][i]) # conversion from matplotlib's reference system
        t0 = abs(tr0[2][0])
        t = abs(tr0[2][i]) - t0 # delta t from first point
        # x equation in A
        A.append([ P[0,0]-x*P[2,0], P[0,0]*t-x*P[2,0]*t, \
                   P[0,1]-x*P[2,1], P[0,1]*t-x*P[2,1]*t, \
                   P[0,2]-x*P[2,2], P[0,2]*t-x*P[2,2]*t ])
        # y equation in A
        A.append([ P[1,0]-y*P[2,0], P[1,0]*t-y*P[2,0]*t, \
                   P[1,1]-y*P[2,1], P[1,1]*t-y*P[2,1]*t, \
                   P[1,2]-y*P[2,2], P[1,2]*t-y*P[2,2]*t ])
 
        # x and y equations in b
        b.append([ x*(P[2,2]*g*t**2/2+P[2,3]) - (P[0,2]*g*t**2/2+P[0,3]) ])
        b.append([ y*(P[2,2]*g*t**2/2+P[2,3]) - (P[1,2]*g*t**2/2+P[1,3]) ])
 
    return np.array(A), np.array(b)


def create_dual_system(tr0, tr1, P):
    # Append on python list is faster than on np.array
    # Conversion is done after the loops
    A = []
    b = []

    t0 = abs(tr0[2][-1]) + (abs(tr1[2][0]) - abs(tr0[2][-1]))/2
    # t0 = set frame

    # first trajectory equations
    for i in range(len(tr0[0])):
        x = tr0[0][i]
        y = abs(tr0[1][i]) # conversion from matplotlib's reference system
        t = abs(tr0[2][i]) - t0 # delta t from first point
        # x equation in A
        A.append([ P[0,0]-x*P[2,0], -P[0,0]*t+x*P[2,0]*t, 0, \
                   P[0,1]-x*P[2,1], -P[0,1]*t+x*P[2,1]*t, 0, \
                   P[0,2]-x*P[2,2], -P[0,2]*t+x*P[2,2]*t, 0 ])
        # y equation in A
        A.append([ P[1,0]-y*P[2,0], -P[1,0]*t+y*P[2,0]*t, 0, \
                   P[1, 1]-y*P[2,1], -P[1,1]*t+y*P[2,1]*t, 0, \
                   P[1,2]-y*P[2,2], -P[1,2]*t+y*P[2,2]*t, 0 ])
        # x and y equations in b
        b.append([ x*(-P[2,2]*g*t**2/2+P[2,3]) + (P[0,2]*g*t**2/2-P[0,3]) ])
        b.append([ y*(-P[2,2]*g*t**2/2+P[2,3]) + (P[1,2]*g*t**2/2-P[1,3]) ])
 
 
    # second trajectory equations
    for i in range(len(tr1[0])):
        x = tr1[0][i]
        y = abs(tr1[1][i]) # conversion from matplotlib's reference system
        t = abs(tr1[2][i]) - t0 # delta t from first point
        # x equation in A
        A.append([ P[0,0]-x*P[2,0], 0, P[0,0]*t-x*P[2,0]*t, \
                   P[0,1]-x*P[2,1], 0, P[0,1]*t-x*P[2,1]*t, \
                   P[0,2]-x*P[2,2], 0, P[0,2]*t-x*P[2,2]*t ])
        # y equation in A
        A.append([ P[1,0]-y*P[2,0], 0, P[1,0]*t-y*P[2,0]*t, \
                   P[1,1]-y*P[2,1], 0, P[1,1]*t-y*P[2,1]*t, \
                   P[1,2]-y*P[2,2], 0, P[1,2]*t-y*P[2,2]*t ])
 
        # x and y equations in b
        b.append([ x*(P[2,2]*g*t**2/2+P[2,3]) - (P[0,2]*g*t**2/2+P[0,3]) ])
        b.append([ y*(P[2,2]*g*t**2/2+P[2,3]) - (P[1,2]*g*t**2/2+P[1,3]) ])
 
    return np.array(A), np.array(b)


def obtain_single_coordinates(p,v,ts):
    # use the physical equations to find the trajectories from starting point
    # and initial velocities

    xt = []
    yt = []
    zt = []
    t0 = abs(ts[0])

    for t in ts:
        delta_t = abs(t) - t0
        xt.append(p[0][0] + delta_t*v[0][0])
        yt.append(p[1][0] + delta_t*v[1][0])
        zt.append(p[2][0] + delta_t*v[2][0] + g*delta_t**2/2)
 
    return xt, yt, zt


def obtain_dual_coordinates(p,v,ts1,ts2):
    
    xt = []
    yt = []
    zt = []
    t0 = abs(ts1[-1]) + (abs(ts2[0]) - abs(ts1[-1]))/2
    # t0 = set frame
    p = p[0]
    v0 = v[0]
    v1 = v[1]

    for t in ts1:
        delta_t = abs(t) - t0
        xt.append(p[0][0] + delta_t*v0[0][0])
        yt.append(p[1][0] + delta_t*v0[1][0])
        zt.append(p[2][0] + delta_t*v0[2][0] + g*delta_t**2/2)
        
    for t in ts2:
        delta_t = abs(t) - t0
        xt.append(p[0][0] + delta_t*v1[0][0])
        yt.append(p[1][0] + delta_t*v1[1][0])
        zt.append(p[2][0] + delta_t*v1[2][0] + g*delta_t**2/2)
 
    return xt, yt, zt


# trajectories = [tr0, tr1, ... , trn]
# tr0 = [ np.array(x), np.array(y), np.array(t) ] 
trajectories = []
points = [] # starting points of sub-trajectoreis
velocities = [] # initial velocities of sub-traj
g = -9.81 # gravitational constant

def _3D_trajectory_estimation(trajectories, P):
    #trajectories = trajectories[1:3]
    print("\n\n3D TRAJ ESTIMATION\n")
    print(trajectories)
    print("\n\nCALIBRATION MATRIX\n")
    np.set_printoptions(suppress = True)
    P = np.array([[ 0.20040696,  0.05544622, -0.00158798, -0.46166368],\
                  [ 0.00574396,  0.02714882, -0.2012131,   0.83807496],\
                  [ 0.0000075,   0.00007497, -0.00000206,  0.00070213]])
    print(P)
    

    # Create and solve system of equations for each sub-trajectory
    for i in range(len(trajectories)):
        A, b = create_single_system(trajectories[i],P)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        # x = [ X0, Vx, Y0, Vy, Z0, Vz ]
        points.append([ x[0], x[2], x[4] ])
        velocities.append([ x[1], x[3], x[5] ])
    
    print(f"\nPOINTS\n{points}")
    print(f"\nVELOCITIES\n{velocities}")

    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(projection ="3d")
    
    # Draw the court
    x1, y1, z1 = [2, 2, 2, 2, 11, 2, 11,2,2], [2, 11, 20, 2, 2, 11, 11,11,11], [0,0,0,0,0,0,0,1.43,2.43]
    x2, y2, z2 = [11, 11, 11, 2, 11, 2, 11,11,11], [2, 11, 20, 20, 20,11,11,11,11], [0,0,0,0,0,2.43,2.43,1.43,2.43]
    for i in range(len(x1)):
        ax.plot([x1[i] , x2[i]], [y1[i] , y2[i]], [z1[i] , z2[i]], c='r')

    # Calculate 3D trajectories and draw them
    for i,tr in enumerate(trajectories):
        if i == 0:
            continue
        xt, yt, zt = obtain_single_coordinates(points[i], velocities[i], tr[2])
        #print(f"\nTRAJECTORY {i} : \nxt: {xt}\nyt: {yt}\nzt: {zt}")
        ax.scatter3D(xt, yt, zt, c='g', marker='o')
        ax.text(xt[0], yt[0], zt[0]+0.1, str(i))
    
    
    """r = 1
    s = 2
    new_points = []
    new_vel = []
    A, b = create_dual_system(trajectories[r],trajectories[s],P)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    new_points.append([ x[0], x[3], x[6] ])
    new_vel.append([ x[1], x[4], x[7] ])
    new_vel.append([ x[2], x[5], x[8] ])
    print(f"\nNEW POINTS : {new_points}")
    print(f"\nNEW VEL : {new_vel}")
    xt, yt, zt = obtain_dual_coordinates(new_points, new_vel, trajectories[r][2], trajectories[s][2])
    #print(f"\nNEW TRAJECTORY {i} : \nxt: {xt}\nyt: {yt}\nzt: {zt}")
    ax.scatter3D(xt, yt, zt, c='b', marker='o')
    ax.text(xt[0], yt[0], zt[0]+0.1, "start")"""

    plt.title("3D plot")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.axes.set_xlim3d(left=0, right=13) 
    ax.axes.set_ylim3d(bottom=0, top=22) 
    ax.axes.set_zlim3d(bottom=0, top=6)

    # show plot
    plt.show()


if __name__ == '__main__':
    trajectories = []
    P = []
    _3D_trajectory_estimation(trajectories,P)
