# -*- coding: utf-8 -*-
import lib601.sm as sm
import lib601.util as util
#import lib601.io as io
from secretMessage import secret
#squarePoints = secret
#squarePoints = [util.Point(1.0, 0.5), util.Point(0.0, 1.0),
               #util.Point(-0.5, 0.5), util.Point(0.0, 0.0)]
import dynamicMoveToPointSkeleton
class FollowFigure(sm.SM):
    def __init__(self,new_list):
        self.startState = 'False'
        self.new_list = new_list
        self.a=0
    def getNextValues(self, state, inp):
        # print(inp)
        distance = self.new_list[self.a].distance(inp.odometry.point())
        if state == 'True':
            if self.a<len(self.new_list)-1:
                self.a+=1
                return 'False', (self.new_list[self.a],inp)
            else:
                return 'True', (self.new_list[self.a],inp)
        if state == 'False' :
            if distance >0.005:
                return 'False', (self.new_list[self.a],inp)
            else:
                return 'True', (self.new_list[self.a],inp)