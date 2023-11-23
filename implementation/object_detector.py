#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

"""Compute the distance between 2 points"""
dist = lambda x1,y1,x2,y2: np.sqrt((x1-x2)**2+(y1-y2)**2)

"""
Given the contour, use the radius of the minimum enclosing circle (MEC)
to compute the difference between the MEC area and the countour area
"""
def get_score(cnt):
    _,radius = cv.minEnclosingCircle(cnt)
    radius = int(radius)
    return np.pi*radius**2 - cv.contourArea(cnt)


class ObjectDetector(object):
    def __init__(self, history: int = 50, thr: float = 50, shadows: bool = False):
        self.obj_detector = cv.createBackgroundSubtractorMOG2(history=history, varThreshold=thr, detectShadows=shadows )
        
    def detect(self, frame, a):
        mask = self.obj_detector.apply(frame)
        contours, _ = cv.findContours(mask[0:a,:], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return mask, contours

    
class BallDetector(object):
    def __init__(self, obj1):
        self.obj_detector_up = obj1   #Object detector above net
        self.out_of_frame = False     #Out-of-frame bool
    
    """
    Given the ball found and the ball predicted
    check whether it is a correct ball
    using the distance between the centers and 
    the minimum distance between the circumferences
    """
    def check_ball(self, x, y, r, x_pred, y_pred, r_pred, farer = False):
        CC_dist = dist(x,y,x_pred,y_pred)
        min_dist = CC_dist-r-r_pred
        if farer==False:
            if CC_dist <= 100 or min_dist <= 50:
                return True
            return False
        else:
            if CC_dist <= 500 or min_dist <= 300:
                return True
            return False
    
    """
    Given the position of the last ball found,
    check whether it is near the borders of the frame
    """
    def check_near_OOF(self, x_last, y_last, shape):
        if x_last < 35 or y_last < 35 or shape[1]-x_last<35 or shape[0]-y_last<35:
            return True
    
    """
    Given the contours, return a sorted list of them
    using the score of the function 'get_score'
    """
    def find_balls(self, contours):
        return sorted(contours, key=get_score)
    
    """
    Given the last 2 ball positions,
    compute an estimate of the next ball position
    """
    def estimate_ball(self, x_1, y_1, x_2, y_2):
        x = 2*x_1-x_2
        y = 2*y_1-y_2
        return x,y
    
    
    """
    Main function detecting the ball
    """    
    def detect(self, frame, tracker):
        ball = None
        ball_history = tracker.get_history('ball')
        if tracker.is_tracked('ball') and tracker.get_history('ball')[-1]!='lost':
            last = ball_history[-1]
            if last == 'OOF':               #Ball was out of frame
                self.out_of_frame = True
                i = 2
                while(ball_history[-i]=='OOF'):
                    i+=1
                (x_last,y_last),r_last,_ = ball_history[-i] #Get the last available ball
            else:                           #Ball was inside frame
                (x_last,y_last),r_last,s = last
    
            print('Last valid ball was: ',last)
            
            ROI = frame[0:400,:,:]      #Region of interest above the net
            mask, contours = self.obj_detector_up.detect(ROI,380)
            
            #Find the ball near the estimated position given the previous ones
            if len(ball_history)>2 and ball_history[-2]!='OOF' and ball_history[-3]!='OOF' and ball_history[-2]!='lost' and ball_history[-3]!='lost' and not self.out_of_frame:
                # Check for a trajectory
                (x_2,y_2),_,_ = ball_history[-2]
                (x_3,y_3),_,_ = ball_history[-3]
                est_ball = self.estimate_ball(x_2, y_2, x_3, y_3)
                if dist(x_last,y_last,est_ball[0],est_ball[1])<10:
                    print('There\'s a trajectory and looking for a ball near the estimated one')
                    est_ball = self.estimate_ball(x_last,y_last,x_2,y_2)    #Estimated ball
                    min_dist = np.inf
                    for cnt in contours:
                        (x,y),radius = cv.minEnclosingCircle(cnt)
                        center = (int(x),int(y))#+400
                        radius = int(radius)
                        if radius < 10: #it is too little
                            continue
        
                        distance = dist(est_ball[0],est_ball[1],center[0],center[1]) #Distance between the ball and the estimated one
                        if distance<50 and (radius in range(r_last-10,r_last+11)) and (np.pi*radius**2 - cv.contourArea(cnt)<1000):
                            if distance < min_dist:     #Looking for the closest contour
                                min_dist = distance
                                ball = (center,radius)
                    if ball is not None:
                        print('Ball found near the estimated one')
                        return ball
            
            print("No ball near the estimated one")

            # Looking for valid contours (not too little)
            valid_contours = []
            for cnt in contours:
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                radius = int(radius)
                if cv.contourArea(cnt) > 100:
                    valid_contours.append(cnt)
            print('Found ',len(valid_contours),'valid contours')
            
            balls = self.find_balls(valid_contours) #List of sorted valid contours
            for cnt in balls:
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                radius = int(radius)
                
                if self.check_ball(center[0],center[1],radius, x_last, y_last, r_last, self.out_of_frame):
                    if self.out_of_frame == True: self.out_of_frame = False
                    good_ball = center,radius
                    if (radius in range(r_last-6,r_last+6)) and (np.pi*radius**2 - cv.contourArea(cnt)<2000):
                        print('Found a ball near the last one')
                        return good_ball    
            
            if not self.out_of_frame:
                if (ball==None):
                    if self.check_near_OOF(x_last, y_last,frame.shape) and len(tracker.get_history('ball'))>2:
                        self.out_of_frame = True
                        print('Ball is Out-Of-Frame')
                        return -2
                    else:
                        print('Ball is lost')
                        return -1
            elif ball == None:  #Ball was OOF and it not found yet
                print('Ball is still OOF')
                return -2
        
        else:   #Ball never tracker or lost
            print('Ricerco la palla sopra la rete...')
            ROI = frame[0:400,:,:]
            mask, contours = self.obj_detector_up.detect(ROI, 370)
            valid_contours = []
            for cnt in contours:
                cv.drawContours(mask, [cnt], -1, (0,255,0), 3)
                if cv.contourArea(cnt) > 200:
                    (x,y),radius = cv.minEnclosingCircle(cnt)
                    center = (int(x),int(y))#+400
                    radius = int(radius)
                    valid_contours.append(cnt)
            
            if len(valid_contours)==0: #No valid contours above the net
                return -1
            
            # There are valid contours - looking for the best ball
            cnt = self.find_balls(valid_contours)[0]
            (x,y),radius = cv.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            return center,radius