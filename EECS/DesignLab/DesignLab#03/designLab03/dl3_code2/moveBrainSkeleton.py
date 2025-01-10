# -*- coding: utf-8 -*-
import os
labPath = os.getcwd()
from sys import path
if not labPath in path:
    path.append(labPath)
print 'setting labPath to', labPath

import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io

# Remember to change the import in dynamicMoveToPointSkeleton in order
# to use it from inside soar
import dynamicMoveToPointSkeleton
reload(dynamicMoveToPointSkeleton)

import ffSkeleton
reload(ffSkeleton)

from secretMessage import secret

# Set to True for verbose output on every step
verbose = False

# Rotated square points
squarePoints = [util.Point(0.5, 0.5), util.Point(0.0, 1.0),
               util.Point(-0.5, 0.5), util.Point(0.0, 0.0)]
temp = 'False'
# Put your answer to step 1 here
class stop(sm.SM):
    def __init__(self):
        self.startState = '0'
    def getNextValues(self, state, inp):
        if 1:
            return ('0',io.Action())

def statemachine():
    return sm.Cascade(ffSkeleton.FollowFigure(squarePoints),dynamicMoveToPointSkeleton.DynamicMoveToPoint())

mySM=sm.Switch(lambda inp:min(inp.sonars)<0.3,stop(),statemachine())


######################################################################
###
###          Brain methods
###
######################################################################

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail = True)
    robot.behavior = mySM

def brainStart():
    robot.behavior.start(traceTasks = robot.gfx.tasks(),
                         verbose = verbose)

def step():
    robot.behavior.step(io.SensorInput()).execute()
    print('iooutput:',io.SensorInput().odometry.point())
    io.done(robot.behavior.isDone())

def brainStop():
    pass

def shutdown():
    pass