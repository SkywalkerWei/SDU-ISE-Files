import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io

import os
labPath = os.getcwd()
from sys import path
if not labPath in path:
    path.append(labPath)
    print 'setting labPath to', labPath

#from boundaryFollower import boundaryFollowerClass
        
class MySMClass(sm.SM):
    startState = 'stop'
    def getNextValues(self, state, inp):
        V01, V02 = inp.analogInputs[1], inp.analogInputs[2]
        rotatev = 0.5 * (5.2 - V01)  # k1 * (V_pot - V01)
        forwardv = 0.1 * (V02 - 7.5)  # k2 * (V02 - V_half)
        action_rotate = ('rotate', io.Action(fvel=0, rvel=rotatev))
        action_stop = ('stop', io.Action(fvel=0, rvel=0))
        action_go = ('go', io.Action(fvel=forwardv, rvel=0))
        if V01 <= 4.9 or V01 >= 5.6:
            return action_rotate
        elif 7.3 <= V02 <= 7.7:  # V_thl <= V02 <= V_thh
            return action_stop
        else:
            return action_go
                
mySM = MySMClass()
mySM.name = 'brainSM'

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=False)

def brainStart():
    robot.behavior = mySM
    robot.behavior.start(robot.gfx.tasks())
    robot.data = []

def step():
    inp = io.SensorInput()
    robot.behavior.step(inp).execute()

def brainStop():
    pass

def shutdown():
    pass