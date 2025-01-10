import math
from soar.io import io
import lib601.util as util
import lib601.sm as sm
import lib601.move as move
import lib601.windows as windows
import lib601.ucSearch as ucSearch

import os
labPath = os.getcwd()
from sys import path
if not labPath in path:
    path.append(labPath)
print 'setting labPath to', labPath

import mapMakerSkeleton as mapMaker
reload(mapMaker)
import lib601.replanner as replanner
reload(replanner)

import soar
import soar.outputs
import soar.outputs.simulator

io.setDiscreteStepLength(0.5)

# Noise variances
noNoise = 0
smallNoise = 0.025
mediumNoise = 0.05
bigNoise = 0.1

# Change noise here
soar.outputs.simulator.SONAR_VARIANCE = lambda mean: noNoise
#soar.outputs.simulator.SONAR_VARIANCE = lambda mean: smallNoise
#soar.outputs.simulator.SONAR_VARIANCE = lambda mean: mediumNoise
#soar.outputs.simulator.SONAR_VARIANCE = lambda mean: bigNoise


######################################################################
###         Setup
######################################################################

mapTestWorld = [0.18, util.Point(2.0, 5.5), (-0.5, 5.5, -0.5, 8.5)]
mazeWorld = [0.15, util.Point(2.0, 0.5), (-0.5, 5.5, -0.5, 5.5)]
dl14World = [0.15, util.Point(3.5, 0.5), (-0.5, 5.5, -0.5, 5.5)]
bigPlanWorld = [0.25, util.Point(3.0, 1.0), (-0.5, 10.5, -0.5, 10.5)]
lizWorld = [0.25, util.Point(9.0, 1.0), (-0.5, 10.5, -0.5, 10.5)]

def useWorld(data):
    global gridSquareSize, goalPoint, xMin, xMax, yMin, yMax
    (gridSquareSize, goalPoint, (xMin, xMax, yMin, yMax)) = data

useWorld(dl14World)


######################################################################
###
###          Brain methods
###
######################################################################

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
    robot.behavior.start()

# this function is called 10 times per second
def step():
    robot.behavior.step(io.SensorInput(cheat = True)).execute()
    io.done(robot.behavior.isDone())

# called when the stop button is pushed
def brainStop():
    pass

# called when brain or world is reloaded (before setup)
def shutdown():
    for w in windows.windowList:
        w.destroy()


