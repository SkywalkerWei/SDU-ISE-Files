import math
from soar.io import io
from lib601.dw import DrawingWindow
import lib601.util as util
import lib601.colors as colors
import lib601.coloredHall as coloredHall
import lib601.seGraphics as seGraphics
import lib601.idealReadings as idealReadings

import os
labPath = os.getcwd()
from sys import path
if not labPath in path:
    path.append(labPath)
print 'setting labPath to', labPath

import lineLocalizeSkeleton as lineLocalize
reload(lineLocalize)

######################################################################
#   Basic parameters
######################################################################


worldPath = 'oneDreal.py'
#worldPath = 'oneDslope.py'
#worldPath = 'oneDdiff.py'
print 'World file is', worldPath

# Where the robot will be in the world
(xMin, xMax) = (0.5, 5.5) 
y = 1.0

# Number of discrete locations
numStates = 120

# Number of discrete observations
numObservations = 30

######################################################################
###  Display the probabilities of the current observation in each state
######################################################################

def drawObsProb(o, numObs, numXSteps, obsModel, w):
    """
    Draw P(o | s) for this particular o = sonar readings on the C{obsWindow}
    """
    y = 0
    unifP = 1.0/numObs
    for x in range(numXSteps):
        w.drawRect((x + 0.2, y + 0.2), (x + 0.8, y + 0.8),
                   color = colors.probToPyColor(obsModel(x).prob(o),
                                                unifP))

######################################################################
#   State Machine Brain
######################################################################

def setup():

    ideal = idealReadings.computeIdealReadings(worldPath, xMin, xMax,
                                               y, numStates, numObservations)

    robot.behavior = \
    lineLocalize.makeLineLocalizer(numObservations, numStates,
                                   ideal, xMin, xMax, y)

    io.setDiscreteStepLength(0.2)    

    # Set up the graphics
    windowWidth = 1000
    # Size of a state in pixels
    stateWidthPixels = windowWidth / float(numStates)
    robot.beliefW = DrawingWindow(numStates*stateWidthPixels,
                                  stateWidthPixels+10,
                                  -0.2, numStates+0.2, -0.2,
                                  1+0.2,'Belief')
    robot.obsW = DrawingWindow(numStates*stateWidthPixels,
                               stateWidthPixels+10,
                               -0.2, numStates+0.2, -0.2,
                               1+0.2,'P(O | S)')
    seGraphics.observationHook =\
      lambda o, obsDist: drawObsProb(o, numObservations, numStates,
                                     obsDist, robot.obsW)
    seGraphics.beliefHook =\
      lambda b: coloredHall.drawBelief(b, robot.beliefW, numStates, False)

    # Start the behavior!
    robot.behavior.start()

def step():
    inp = io.SensorInput(cheat = True)
    # Compute next action (will draw belief state)
    act = robot.behavior.step(inp)
    # Draw true state
    trueState = util.clip(int(round(numStates * (inp.odometry.x - xMin) /\
                                    (xMax - xMin))),
                          0, numStates-1)
    robot.beliefW.drawRect((trueState+0.3, 0.3), (trueState+0.7,0.7),'gold')
    # Execute action
    act.execute()

def shutdown():
    robot.beliefW.destroy()
    robot.obsW.destroy()

