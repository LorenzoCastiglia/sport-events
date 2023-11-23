#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv

class VideoController(object):
    def __init__(self):
        self.paused = False
    
    def command(self, cmd):
        if cmd==ord('p') or cmd==ord('P'):    #PAUSE/PLAY
            if not self.paused: 
                self.paused = True
                key = cv.waitKey(0)
                return self.command(key)
            else:
                self.paused = False
                return 1
        elif cmd==ord('m') or cmd==ord('M'):  #FRAME BY FRAME
            if self.paused:
                return 0
        elif cmd==ord('q') or cmd==ord('Q'):  #CLOSE EVERYTHING
            return cmd
        elif cmd==ord('k') or cmd==ord('K'):  #SKIP 5 SECONDS
            return 125
        elif cmd==ord('s') or cmd==ord('S'):  #START/STOP VIDEO TRACKING
            return cmd
        return 1
                
        