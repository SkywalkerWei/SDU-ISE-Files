import math
from soar.io import io
import lib601.util as util
import lib601.sm as sm
import lib601.windows as windows
import lib601.ucSearch as ucSearch
reload(ucSearch)

import os
labPath = os.getcwd()
from sys import path
if not labPath in path:
    path.append(labPath)
print 'setting labPath to', labPath

import soar
import soar.outputs
import soar.outputs.simulator
import move
reload(move)
import replannerRace as replanner
reload(replanner)

# Import the following file from lib601 to use our implementation
# instead of yours.
import mapMakerSkeleton as mapMaker
reload(mapMaker)

# Parameters in motion controller
move.MoveToDynamicPoint.forwardGain = 0.5
move.MoveToDynamicPoint.rotationGain = 1.0
move.MoveToDynamicPoint.angleEps = 0.05

# Noise variances
noNoise = 0
smallNoise = 0.025
mediumNoise = 0.05
bigNoise = 0.1

# Change noise here
soar.outputs.simulator.SONAR_VARIANCE = lambda mean: mediumNoise

# Leave this alone
io.setDiscreteStepLength(0.5)

######################################################################
###         Setup
######################################################################

mapTestWorld = [0.18, util.Point(2.0, 5.5), (-0.5, 5.5, -0.5, 8.5)]
raceWorld = [0.18, util.Point(2.0, 5.5), (-0.5, 5.5, -0.5, 8.5)]
mazeWorld = [0.15, util.Point(2.0, 0.5), (-0.5, 5.5, -0.5, 5.5)]
frustrationWorld = [0.15, util.Point(3.5, 0.5), (-0.5, 5.5, -0.5, 5.5)]
bigPlanWorld = [0.25, util.Point(3.0, 1.0), (-0.5, 10.5, -0.5, 10.5)]
lizWorld = [0.25, util.Point(9.0, 1.0), (-0.5, 10.5, -0.5, 10.5)]

def useWorld(data):
    global gridSquareSize, goalPoint, xMin, xMax, yMin, yMax
    (gridSquareSize, goalPoint, (xMin, xMax, yMin, yMax)) = data

useWorld(raceWorld)

######################################################################
###
###          Brain methods
###
######################################################################

import time

# this function is called when the brain is (re)loaded
def setup():
    mapper = mapMaker.MapMaker(xMin, xMax, yMin, yMax, gridSquareSize)
    replannerSM = replanner.ReplannerWithDynamicMap(goalPoint)
    robot.behavior = \
       sm.Cascade(sm.Parallel(sm.Cascade(sm.Parallel(mapper, sm.Wire()),
                                         replannerSM),
                              sm.Wire()), 
                  move.MoveToDynamicPoint())

# this function is called when the start button is pushed
def brainStart():
    robot.count = 0
    robot.startTime = time.time()
    robot.behavior.start()

# this function is called 10 times per second
def step():
    robot.count += 1
    inp = io.SensorInput(cheat = True)
    robot.behavior.step(inp).execute()
    io.done(inp.odometry.point().isNear(goalPoint, gridSquareSize))

# called when the stop button is pushed
def brainStop():
    stopTime = time.time()
    print 'Total steps:', robot.count
    print 'Elapsed time in seconds:', stopTime - robot.startTime

# called when brain or world is reloaded (before setup)
def shutdown():
    for w in windows.windowList:
        w.destroy()


