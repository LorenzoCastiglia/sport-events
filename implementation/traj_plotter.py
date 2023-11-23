#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import numpy as np
import matplotlib.pyplot as plt

"""Compute the distance"""
dist = lambda A,B: np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)
"""Compute the mid-point"""
x_mid_point = lambda x1,x2: (x1+x2)/2
"""Compute the distance of a point from a line"""
proximity = lambda xA,yA,xB,yB,xP,yP: np.abs( ((yA-yB)/(xA-xB))*xP-yP+(yB-((yA-yB)/(xA-xB))*xB) ) if xA!=xB else np.abs(xP-xA)
"""Compute the distance of a point from a parabola"""
proximity_parabola = lambda A,B,C,x,y: np.abs(A*x**2+B*x+C-y)
"""Compute the vertex of a parabola"""
vertex = lambda A,B,C: A*(-B/(2*A))**2+B*(-B/(2*A))+C

"""
Draw the court on the figure
"""
def draw_court(fig):
    with open('Sources/2D points.txt', 'r') as f:
        points2D = np.array([ast.literal_eval(line) for line in f])
    (xO,yO)=points2D[0]
    (xA,yA)=points2D[4]
    (xB,yB)=points2D[5]
    (xE,yE)=points2D[8]
    A = (yB-yA)/(xB-xA)
    B = yO-A*xO
    C = (yE-yB)/(xE-xB)
    D = yB-C*xB
    x = (D-B)/(A-C)
    y = A*x+B
    points2D = np.insert(points2D,9,(x,y),axis=0)
    
    plt.plot([points2D[0][0],points2D[9][0]],[-points2D[0][1],-points2D[9][1]],'-k',figure=fig)
    plt.plot([points2D[0][0],points2D[4][0]],[-points2D[0][1],-points2D[4][1]],'-k',figure=fig)
    plt.plot([points2D[4][0],points2D[5][0]],[-points2D[4][1],-points2D[5][1]],'-k',figure=fig)
    plt.plot([points2D[5][0],points2D[9][0]],[-points2D[5][1],-points2D[9][1]],'-k',figure=fig)
    plt.plot([points2D[1][0],points2D[8][0]],[-points2D[1][1],-points2D[8][1]],'-k',figure=fig)
    plt.plot([points2D[2][0],points2D[7][0]],[-points2D[2][1],-points2D[7][1]],'-k',figure=fig)
    plt.plot([points2D[3][0],points2D[6][0]],[-points2D[3][1],-points2D[6][1]],'-k',figure=fig)
    plt.plot([points2D[2][0],points2D[11][0]],[-points2D[2][1],-points2D[11][1]],'-k',figure=fig)
    plt.plot([points2D[11][0],points2D[12][0]],[-points2D[11][1],-points2D[12][1]],'-k',figure=fig)
    plt.plot([points2D[7][0],points2D[12][0]],[-points2D[7][1],-points2D[12][1]],'-k',figure=fig)
    plt.plot([points2D[10][0],points2D[13][0]],[-points2D[10][1],-points2D[13][1]],'-k',figure=fig)

"""
Given two parabolae, compute the intersection
"""
def intersection(traj1, traj2, dir1, dir2, lasty):
    (a1,b1,c1) = traj1
    (a2,b2,c2) = traj2
    A = a1-a2
    B = b1-b2
    C = c1-c2
    if A==0:
        if B==0:
            print('Error: no intersection')
        else:
            x = -C/B
            y = a1*(x**2)+b1*x+c1
    else:
        x1 = (-B+np.sqrt(B**2-4*A*C))/(2*A)
        x2 = (-B-np.sqrt(B**2-4*A*C))/(2*A)
        
        if dir1!=dir2:
            if dir1=='l':
                x = min(x1,x2)
                y = a1*(x**2)+b1*x2+c1
            else:
                x = max(x1,x2)
                y = a1*(x**2)+b1*x+c1
        else:
            y1 = a1*(x1**2)+b1*x1+c1
            y2 = a1*(x2**2)+b1*x2+c1
            if np.abs(y1-lasty)<np.abs(y2-lasty):
                x = x1
                y = y1
            else:
                x = x2
                y = y2
    return (x,y)

"""
Given a line and a parabola,
compute their intersection
"""
def line_parabola_intersection(line,parabola,x_p):
    (A,B,C)=parabola
    (M,Q)=line
    x_1 = (-(B-M)+np.sqrt((B-M)**2-4*A*(C-Q)))/(2*A)
    x_2 = (-(B-M)-np.sqrt((B-M)**2-4*A*(C-Q)))/(2*A)
    if np.abs(x_1-x_p)<np.abs(x_2-x_p):
        return x_1
    else:
        return x_2       

class Plotter(object):
    def __init__(self):
        self.stop_points = []
        self.attack = False
    
    """
    Check if the trajectory is valid:
        at least 5 ball points
    """
    def check_trajectory(self, traj):
        if len(traj)<5:
            return False
        length = 0
        for t in traj:
            if t != 'lost':
                length += 1
        if length < 5:
            return False
        return True
    
    """Clean the trajectory from OOF and lost"""  
    def traj_cleaner(self, traj):
        traj_copy = traj.copy()
        for i in range(len(traj_copy)-1,-1,-1):
            if traj_copy[i]=='lost' or traj_copy[i]=='OOF':
                del traj_copy[i]
        return traj_copy
    
    """
    Given the full trajectory of the ball,
    divide it into sub-trajectory among the contacts with players
    """
    def gen_sub_trajectories(self, full_traj):
        print("Looking for sub-trajectories")
        sub_trajs = []
        last_x = 0
        last_y = 0

        x = np.array([full_traj[0][0][0],full_traj[1][0][0],full_traj[2][0][0]])
        y = np.array([-full_traj[0][0][1],-full_traj[1][0][1],-full_traj[2][0][1]])
        ts_list = np.array([-full_traj[0][2],-full_traj[1][2],-full_traj[2][2]])
        
        for i in range(3,len(full_traj)):
            xp = full_traj[i][0][0]
            yp = -full_traj[i][0][1]
            ts = full_traj[i][2]
            
            if len(y)>2 and len(x)>2 and (y[-2]-y[-3])<0 and (y[-1]-y[-2])<0 \
                and (((yp-y[-1])>0 and proximity(x[-1],y[-1],x[-2],y[-2],xp,yp)>=4) \
                or (proximity(x[-1],y[-1],x[-2],y[-2],xp,yp)>100)):
                
                self.stop_points.append([xp,yp])
                print('New parabola')
                traj = np.polyfit(x, y, 2)  #Compute the parabola parameters given the points
                
                if (((np.abs(traj[0])<0.5 and vertex(traj[0], traj[1], traj[2])<-100) if len(sub_trajs)==0 else True) and traj[0]<0):
                    
                    if len(sub_trajs)<1:
                        first_point = [x[0],y[0]]
                        direction = 'l' if x[-1]<x[0] else 'r'
                    else:
                        direction = 'l' if x[-1]<x[0] else 'r'
                        (x_i, y_i) = intersection(sub_trajs[-1][1],traj,sub_trajs[-1][2],direction, y[0])
                        if (x_i>last_x and sub_trajs[-1][2]=='r') or (x_i<last_x and sub_trajs[-1][2]=='l'):
                            first_point = (x_i, y_i)
                        elif np.abs(last_x-x_i)<20:
                            first_point = (x_i, y_i)
                        else:
                            print('Attack')
                            first_point = [x[0],y[0]]
                            sub_trajs.append([last_x,-last_y])
                    sub_trajs.append([first_point,traj,direction,x,y,ts_list])
                else:
                    print('Ball launch for service')
                
                #New parabola
                last_x = x[-1]
                last_y = y[-1]
                x = np.array([xp])
                y = np.array([yp])
                ts_list = np.array([ts])
            else:
                x = np.append(x,xp)
                y = np.append(y,yp)
                ts_list = np.append(ts_list,ts)
        
        # Remaining points
        if len(x)>2:
            traj = np.polyfit(x, y, 2)
            if len(sub_trajs)<1:
                first_point = [x[0],y[0]]
                direction = 'l' if x[-1]<x[0] else 'r'
            else:
                direction = 'l' if x[-1]<x[0] else 'r'
                (x_i, y_i) = intersection(sub_trajs[-1][1],traj,sub_trajs[-1][2],direction, y[0])
                if (x_i>last_x and sub_trajs[-1][2]=='r') or (x_i<last_x and sub_trajs[-1][2]=='l'):
                    first_point = (x_i, y_i)
                elif np.abs(last_x-x_i)<20:
                    first_point = (x_i, y_i)
                else:
                    print('Final Attack')
                    first_point = [x[0],y[0]]
                    sub_trajs.append([last_x,-last_y])
                    
            sub_trajs.append([first_point,traj,direction, x,y,ts_list])
            sub_trajs.append([x[-1],-y[-1]])
        return sub_trajs 
    
    """
    Handle the attack situation,
    linking the 2 trajectories 
    or creating the last one
    """
    def attack_handler(self, trajs, i, fig):
        print("Attack handler")
        if len(trajs)==3:   #There is the next trajectory
            #print("QUI")
            #t1 = trajs[0]
            sp = trajs[1]
            t2 = trajs[2]
            
            if t2[2]=='l':
                
                mid_point_x = t2[0][0]+100
                mid_point_y = t2[1][0]*(mid_point_x**2)+t2[1][1]*mid_point_x+t2[1][2]
                x_s = [sp[0],mid_point_x]
                y_s = [-sp[1],mid_point_y]
                plt.plot(x_s,y_s,"-b", figure=fig)
                plt.text((sp[0]+mid_point_x)/2+20,(-sp[1]+mid_point_y)/2-20, str(i),figure=fig)
                x_s = np.linspace(mid_point_x, t2[0][0], 50)
                poly = np.poly1d(t2[1])
                plt.plot(x_s,poly(x_s),'-b',figure=fig)
                
            elif t2[2]=='r':
                mid_point_x = t2[0][0]-100
                mid_point_y = t2[1][0]*(mid_point_x**2)+t2[1][1]*mid_point_x+t2[1][2]
                x_s = [sp[0],mid_point_x]
                y_s = [-sp[1],mid_point_y]
                plt.plot(x_s,y_s,"-b", figure=fig)
                plt.text((sp[0]+mid_point_x)/2-20,(-sp[1]+mid_point_y)/2-20, str(i),figure=fig)
                x_s = np.linspace(mid_point_x, t2[0][0], 50)
                poly = np.poly1d(t2[1])
                plt.plot(x_s,poly(x_s),'-b',figure=fig)
                
        else:       
            pass
    
    """
    Plot the complete ball trajectory
    """
    def plot(self, trajectory, shape):
        self.stop_points = []
        _3d_traj = []   #Output for 3D plotting
        if self.check_trajectory(trajectory):
            print('Plot the trajectory')
            fig = plt.figure("TRAJECTORY")
            ax = plt.axes()
            ax.set_xlim(0,shape[1]+200)
            ax.set_ylim(-shape[0]-50,0)
            
            #Plotting the points
            for t in trajectory:
                if t != 'OOF' and t!='lost':
                    (x,y)=t[0]
                    plt.plot(x,-y, '.:r', figure=fig)
            
            sub_trajs = self.gen_sub_trajectories(self.traj_cleaner(trajectory))
            i = 0
            while len(sub_trajs)==2 and sub_trajs[0][1][0]>0: #Lancio palla
                i += 1
                sub_trajs = self.gen_sub_trajectories(self.traj_cleaner(trajectory[i:]))
            
            print('Found',len(sub_trajs),'sub-trajectories')
            #print(sub_trajs)
            if len(sub_trajs)>2:
                for i in range(0,len(sub_trajs)-1):
                    st = sub_trajs[i]
                    if len(st)==2:  #it's a stop point
                        if len(sub_trajs[i+1])!=2: #it's not a point
                            self.attack_handler([sub_trajs[i-1],st,sub_trajs[i+1]],i,fig)
                        continue
                    
                    start = st[0][0]
                    if i==len(sub_trajs)-2:
                        stop = sub_trajs[i+1][0]
                        j=0
                        while(trajectory[j]=='OOF' or trajectory[j]=='lost' or trajectory[j][0][0]!=sub_trajs[i+1][0] or trajectory[j][0][1]!=sub_trajs[i+1][1]):
                            j+=1
                        j+=1
                        while(trajectory[j]=='OOF' or trajectory[j]=='lost'):
                            j+=1
                            if j==len(trajectory)-1:
                                break
                        
                        _3d_traj.append([st[3],st[4],st[5]])
                        
                    elif len(sub_trajs[i+1])==2:
                        j=0
                        while(trajectory[j]=='OOF' or trajectory[j]=='lost' or trajectory[j][0][0]!=sub_trajs[i+1][0] or trajectory[j][0][1]!=sub_trajs[i+1][1]):
                            j+=1
                        j+=1
                        while(trajectory[j]=='OOF' or trajectory[j]=='lost'):
                            j+=1
                        stop = sub_trajs[i+1][0]
                        _3d_traj.append([st[3],st[4],st[5]])
                    else:
                        stop = sub_trajs[i+1][0][0]
                        _3d_traj.append([st[3],st[4],st[5]])
                    
                    xp = np.linspace(start, stop, 200)
                    poly = np.poly1d(st[1])
                    plt.plot(xp,poly(xp),'-b',figure=fig)
                    
                    a = st[1][0]
                    b = st[1][1]
                    c = st[1][2]
                    plt.text(-b/(2*a)-10,vertex(a,b,c)+15,str(i) if i!=0 else "Serve")
                
            else: #only 1 trajectory (serve)
                print("Only the service")
                d = sub_trajs[0][2]
                stop = sub_trajs[1][0]-400 if d=='l' else sub_trajs[1][0]+400
                xp = np.linspace(sub_trajs[0][0][0], stop, 100)
                poly = np.poly1d(sub_trajs[0][1])
                plt.plot(xp,poly(xp),'-b',figure=fig)
                _3d_traj.append([sub_trajs[0][3],sub_trajs[0][4],sub_trajs[0][5]])
                
                a = sub_trajs[0][1][0]
                b = sub_trajs[0][1][1]
                c = sub_trajs[0][1][2]
                plt.text(-b/(2*a)-10,vertex(a,b,c)+15,"Serve")
                
            draw_court(fig)
            plt.show()
            return _3d_traj
    