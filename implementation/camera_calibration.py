#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import ast
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

points2D = []

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        points2D.append((x,y))

class CameraCalibration(object):
    def __init__(self):
        self.points2D = []
        self.points3D = []
        self.calibration_matrix = np.zeros(shape=(3,4))
        
    def draw_line(self,points,frame):
        cv.line(frame,points[0],points[9],(0,0,255),2)
        cv.line(frame,points[0],points[4],(0,0,255),2)
        cv.line(frame,points[4],points[5],(0,0,255),2)
        cv.line(frame,points[5],points[9],(0,0,255),2)
        cv.line(frame,points[1],points[8],(0,0,255),2)
        cv.line(frame,points[2],points[7],(0,0,255),2)
        cv.line(frame,points[3],points[6],(0,0,255),2)
        cv.line(frame,points[10],points[11],(0,0,255),2)
        cv.line(frame,points[11],points[12],(0,0,255),2)
        cv.line(frame,points[12],points[13],(0,0,255),2)
        cv.line(frame,points[10],points[13],(0,0,255),2)
        cv.imshow("court",frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
    
    def court_plot(self):
        fig = plt.figure(figsize=(20,20))
        ax = plt.axes(projection='3d')
        ax.set_zlim(0,5)
        for p in self.points3D:
            ax.scatter(p[0],p[1],p[2],marker='o')
        couples = [[0,4],[4,5],[5,9],[0,9],[1,8],[2,7],[3,6],[10,11],[11,12],[12,13],[10,13]]
        for c in couples:
            A = self.points3D[c[0]]
            B = self.points3D[c[1]]
            ax.plot([A[0],B[0]],[A[1],B[1]],[A[2],B[2]])
        plt.show()
    
    def calibrate(self,frame, use_points = True):
        # Edge detection
        #cv.namedWindow("court")
        court_img = frame
        if not use_points:
            cv.setMouseCallback('court', click_event)
            while True:
                # both windows are displaying the same img
                for (x,y) in points2D:
                    cv.circle(court_img,(x,y),1,(0,0,255),10)
                cv.imshow("court", court_img)
                if cv.waitKey(1) & 0xFF == ord("m"):
                    break
            cv.destroyAllWindows()
            cv.waitKey(1)
            self.points2D = points2D
            with open('Sources/2D points.txt', 'w') as f:
                for p in self.points2D:
                    f.write(str(p)+'\n')
        else:
            with open('Sources/2D points.txt', 'r') as f:
                self.points2D = np.array([ast.literal_eval(line) for line in f])
            #print(self.points2D)
        print('Shape:',frame.shape)
        for p in self.points2D:
            p[1]=frame.shape[0]-p[1]
            print(p)
        
        (xO,yO)=self.points2D[0]
        (xA,yA)=self.points2D[4]
        (xB,yB)=self.points2D[5]
        (xE,yE)=self.points2D[8]
        A = (yB-yA)/(xB-xA)
        B = yO-A*xO
        C = (yE-yB)/(xE-xB)
        D = yB-C*xB
        x = (D-B)/(A-C)
        y = A*x+B
        self.points2D = np.insert(self.points2D,9,(x,y),axis=0)
        print(self.points2D)
        with open('Sources/3D points.txt', 'r') as f:
            self.points3D = np.array([ast.literal_eval(line) for line in f])
        #print(points3D)
        #self.court_plot()
        
        #A = np.atleast_2d(self.calibration_matrix[0,:]).transpose()
        #B = np.atleast_2d(self.calibration_matrix[1,:]).transpose()
        #C = np.atleast_2d(self.calibration_matrix[2,:]).transpose()
        
        #p = np.concatenate([A,B,C],axis=0)
        #print(p)
        M = np.zeros(shape=(2*len(self.points3D),12))
        for i in range(len(self.points3D)):
            p3d = self.points3D[i]
            p2d = self.points2D[i]
            a_x = np.array([-p3d[0],-p3d[1],-p3d[2],-1,0,0,0,0,p2d[0]*p3d[0],p2d[0]*p3d[1],p2d[0]*p3d[2],p2d[0]])
            a_y = np.array([0,0,0,0,-p3d[0],-p3d[1],-p3d[2],-1,p2d[1]*p3d[0],p2d[1]*p3d[1],p2d[1]*p3d[2],p2d[1]])
            M[2*i]=a_x
            M[2*i+1]=a_y
        M = M.astype(int)
        #print(M)
        _,_,V = np.linalg.svd(M)
        # print(U)
        # print(S)
        #print(V.shape)
        p = V.transpose()[:,-1]
        #print(p.shape)
        #print(self.calibration_matrix)
        for i in range(len(p)):
            self.calibration_matrix[int(np.floor(i/4)),int(np.mod(i,4))]=p[i]
        #print(self.calibration_matrix)
        # print(self.points2D[2])
        # print(np.atleast_2d(np.append(self.points3D[0],1)).transpose())
        # tmp = self.calibration_matrix.dot(np.atleast_2d(np.append(self.points3D[2],1)).transpose())
        # tmp = tmp/tmp[2]
        # print(tmp)
        return self.calibration_matrix

        