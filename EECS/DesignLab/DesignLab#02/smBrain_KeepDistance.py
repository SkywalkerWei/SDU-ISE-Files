import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io

class MySMClass(sm.SM):
    startState = None
    def getNextValues(self, state, inp):
        desiredDistance = 0.5
        currentDistance0 = inp.sonars[0]
        currentDistance1 = inp.sonars[1]
        currentDistance2 = inp.sonars[2]
        currentDistance3 = inp.sonars[3]
        currentDistance4 = inp.sonars[4]
        currentDistance5 = inp.sonars[5]
        currentDistance6 = inp.sonars[6]
        currentDistance7 = inp.sonars[7]
        a = 0
        if a == 0:
            if ((currentDistance3 + currentDistance4)/2<0.5):
                return (state, io.Action(fvel=0.1*((currentDistance3 + currentDistance4)/2-desiredDistance),rvel=0))
            else:
                a = 1
                return (state, io.Action(fvel=0.1, rvel=0))
        if a == 1:
            if ((currentDistance3 + currentDistance4)/2>0.5):
                return (state, io.Action(fvel=0.1*((currentDistance3 + currentDistance4)/2-desiredDistance),rvel=0))
            else:
                a = 0
                return (state, io.Action(fvel=0.1, rvel=0))




mySM = MySMClass()
mySM.name = 'brainSM'

######################################################################
###
###          Brain methods
###
######################################################################

def plotSonar(sonarNum):
    robot.gfx.addDynamicPlotFunction(y=('sonar'+str(sonarNum),
                                        lambda: 
                                        io.SensorInput().sonars[sonarNum]))

# this function is called when the brain is (re)loaded
def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=True,                                 sonarMonitor=True)
    robot.behavior = mySM

# this function is called when the start button is pushed
def brainStart():
    robot.behavior.start(traceTasks = robot.gfx.tasks())

# this function is called 10 times per second
def step():
    inp = io.SensorInput()
    robot.behavior.step(inp).execute()
    io.done(robot.behavior.isDone())

# called when the stop button is pushed
def brainStop():
    pass

# called when brain or world is reloaded (before setup)
def shutdown():
    pass
