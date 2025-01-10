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
    startState="turn"
    def getNextValues(self, state, inp):
        V_Location, V_Light = inp.analogInputs[1], inp.analogInputs[2]
        V_diffLocation, V_diffLight = 5.0 - V_Location, V_Light - 7.5  # V_Base = 5.0, V_half = 7.5
        k_Location, k_Light = 0.5, 1
        action_stop = ("light", io.Action(fvel=0, rvel=0))
        if state == "turn":
            if V_Location == 5.0:  # V_Base
                return action_stop
            return ("turn", io.Action(fvel=0, rvel=k_Location * V_diffLocation))
        if state == "light":
            if 7 <= V_Light <= 8:  # V_thl <= V_Light <= V_thh
                return ("turn", io.Action(fvel=0, rvel=0))
            elif V_Light < 7:
                return('light',io.Action(fvel = -k_Light*V_diffLight, rvel = 0))
            elif V_Light > 8:
                return('light',io.Action(fvel = k_Light*V_diffLight, rvel = 0))
            
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