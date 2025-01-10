import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io

class MySMClass(sm.SM):
    offset = 0.1
    def getNextValues(self, state, inp):
        return (state, io.Action(fvel = 0.0, rvel = 0.0))

mySM = MySMClass()
mySM.name = 'brainSM'

def settleTime(samples):
    epsilon = (max(samples)-min(samples))*0.05
    lastUnchangedIdx = None
    for i in range(len(samples)):
        if abs(samples[i]-samples[0]) >= epsilon:
            lastUnchangedIdx = i-1
            break
    lastBadIdx = None
    for i in range(len(samples)-2, -1, -1):
        if abs(samples[i]-samples[-1]) > epsilon:
            lastBadIdx = i
            break
    if lastUnchangedIdx is not None and lastBadIdx is not None:
        return lastBadIdx - lastUnchangedIdx + 1
    

######################################################################
###
###          Brain methods
###
######################################################################

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=False)
    startTheta = io.SensorInput().odometry.theta
    # static plot of robot angle (from start) vs time
    robot.gfx.addStaticPlotFunction(
        y = ('neck', lambda inp: inp.analogInputs[0]))

def brainStart():
    robot.behavior = mySM
    robot.behavior.start(robot.gfx.tasks())
    robot.data = []

def step():
    inp = io.SensorInput()
    robot.behavior.step(inp).execute()
    neck = inp.analogInputs[0]
    robot.data.append(neck)

def brainStop():
    print 'settle time: ', settleTime(robot.data)

def shutdown():
    pass
