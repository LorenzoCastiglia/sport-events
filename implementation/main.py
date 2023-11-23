#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from object_detector import ObjectDetector, BallDetector
from video_controller import VideoController
from tracking import Tracker
from traj_plotter import Plotter
from camera_calibration import CameraCalibration
from time import sleep
from traj_estimation_3D import _3D_trajectory_estimation
import argparse

formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)
parser = argparse.ArgumentParser(description="2D and 3D ball trajectory estimation\
                                 and visualization of volleyball videos",\
                                 formatter_class=formatter)
parser.add_argument("-v", "--video_id", type=str, default="1",\
                    help="Select video id from 1 to 7, default is 1", metavar='[1-7]')
parser.add_argument("-f", "--fps", type=int, default=25,\
                    help="Specify the fps of the video, default is 25", metavar="[1-240]")
args = parser.parse_args()
video_id = args.video_id
fps = args.fps
video = cv.VideoCapture("Sources/Actions/0209" + video_id + ".mp4")

obj_detector_up = ObjectDetector(history=20, thr=100, shadows=False )
tracker = Tracker()
videoControl = VideoController()
cameraCalibration = CameraCalibration()
plotter = Plotter()
frame_cnt = 0
command = 1
skip = 0
analyze = True

def check_end_action(history):
    for i in range(1,41 if len(history)>=40 else len(history)+1):
        if history[-i]!='OOF' and history[-i]!='lost': 
            return False
    return True

def print_commands():
    cv.rectangle(frame,(35,15),(480,190),(0,0,0),-1)
    cv.putText(frame,"COMMANDS", (50,50),cv.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    cv.putText(frame,"p: PLAY/PAUSE", (50,75),cv.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    cv.putText(frame,"m: FRAME BY FRAME", (50,100),cv.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    cv.putText(frame,"k: SKIP 5 SECONDS", (50,125),cv.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    cv.putText(frame,"s: START/STOP ANALYSIS", (50,150),cv.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    cv.putText(frame,"q: QUIT", (50,175),cv.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)


while True:
    sleep(0.01)
    ret, frame = video.read()
    #print("QUI")
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if skip!=0: 
        skip -= 1
        continue
    
    frame_cnt += 1
    cv.namedWindow("frame", cv.WINDOW_NORMAL)

    cv.resizeWindow("frame", 710, 400)
    
    if frame_cnt==1:
        calibration_matrix = cameraCalibration.calibrate(frame)
        #print(frame.shape)

    if analyze:
        frame_cpy = frame.copy()
        y_cut = 400
        mask_ROI = [[0,frame.shape[0]],[0,y_cut],[348,y_cut],[348,320],[353,320],[353,1000],[1463,1000],[1463,330],[1467,330],[1467,450],[1550,450],[1640,370],[1640,350],[frame.shape[1],350],[frame.shape[1],frame.shape[0]],[0,frame.shape[0]]]
        cv.fillConvexPoly(frame_cpy, np.array(mask_ROI, dtype=np.int32),(0,0,0))
        cv.circle(frame, (frame.shape[1]-30,30),20,(0,255,0),-1)
        ball_detector = BallDetector(obj_detector_up)
        ball = ball_detector.detect(frame_cpy,tracker)
        if ball==-2:
            cv.putText(frame,"OutOfFrame", (int(frame.shape[1]/2),frame.shape[0]-50), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            tracker.track_ball('OOF')
        elif ball==-1:
            if tracker.is_tracked('ball'):
                tracker.track_ball('lost')
                cv.putText(frame,"NO BALL", (int(frame.shape[1]/2),frame.shape[0]-50), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                
        else:
            if frame_cnt>1:
                print(ball)
                #if ball[1]<30:
                tracker.track_ball(ball, frame_cnt/fps)
                
                cv.circle(frame,ball[0],ball[1],(0,0,255),10)
            
        if frame_cnt>1:
            if tracker.is_tracked('ball') and check_end_action(tracker.get_history('ball')):
                print('Azione finita')
                points_list = plotter.plot(tracker.get_history('ball'),frame.shape)
                #print("Points list:", points_list)
                if points_list is not None:
                    _3D_trajectory_estimation(points_list,calibration_matrix)
                tracker.delete('ball')
                cv.putText(frame,"End of Action", (int(frame.shape[1]/2),frame.shape[0]-50), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            hst = tracker.get_history('ball')[:-1]
            for c in hst:
                if c!='OOF' and c!='lost':
                    cv.circle(frame, c[0],1,(255,0,0),10)
    else:
        tracker.delete('ball')
    
    print_commands()
    cv.imshow("frame", frame)
    #cv.imshow("mask", frame_cpy)
    
    #print(command)
    key = cv.waitKey(command) & 0xFF
    command = videoControl.command(key)
    if command == ord('q') or command == ord('Q'):
        break
    if command == ord('s') or command==ord('S'):
        if analyze: plotter.plot(tracker.get_history('ball'),frame.shape)
        analyze = not analyze
        tracker.delete('ball')
        command = 1
        
    skip = command if command > 1 else skip
    print("-----------------------------------------")
    
cv.destroyAllWindows()
cv.waitKey(1)

video.release()
print("FINE")