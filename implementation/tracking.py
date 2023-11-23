#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Tracker(object):
    def __init__(self):
        self.tracked_elements = {}
        self.count = 0
        self.mean_radius = 0
    
    def track_ball(self, ball, ts = 0.0):
        if not self.tracked_elements or not self.is_tracked('ball'): #New ball
            self.count = 1
            self.mean_radius = ball[1]
            self.tracked_elements['ball'] = ([[ball[0],ball[1],ts]])
        else:
            if ball == 'OOF':
                self.tracked_elements['ball'].append('OOF')
            elif ball == 'lost':
                self.tracked_elements['ball'].append('lost')
            else:
                self.mean_radius = (self.mean_radius*self.count+ball[1])/(self.count+1)
                self.count += 1
                self.tracked_elements['ball'].append([ball[0],ball[1],ts])
    
    def get_mean_radius(self):
        return self.mean_radius
    
    def get_history(self,id):
        return self.tracked_elements[id] if self.is_tracked('ball') else []
        
    def is_tracked(self, id):
        return id in self.tracked_elements.keys()
    
    def delete(self,id):
        self.tracked_elements.pop(id, None)
        
        
    