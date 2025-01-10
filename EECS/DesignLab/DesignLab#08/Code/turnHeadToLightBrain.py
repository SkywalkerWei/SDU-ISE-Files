#
# File:   turnHeadToLightBrain.py
# Date:   24-Oct-11
# Author: 6.01 Staff
#
# Robot brain for Design Lab 8
#
# Connect the photoresistor voltages to analog inputs 2 (analogInput[1]) and 3 (analogInput[2])
# Connect AOUT (voltage) to the positive terminal of the motor.
# Connect your V_X circuit to the negative terminal of the motor.
#
# This brain file uses a proportional controller to drive the motor to point the robot's
# photoresistive eyes towards an incident light source.
#
# A plot is presented, showing the photoresitor voltage difference vs time step

import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
import lib601.fr as fr
from soar.io import io

#-----------------------------------------------------------------------------
# robot brain state machine

class MySM(sm.SM):
    def getNextValues(self, state, inp):
        left = inp.analogInputs[1]
        right = inp.analogInputs[2]
        gain=1.0			# try varying this to increase speed
        ctrl_out=5.0+gain*(left-right)
        nextstate=ctrl_out			
        v_out = ctrl_out		# change to v_out = state for extra delay
        return (nextstate, io.Action(fvel = 0.0, rvel = 0.0, voltage = v_out))

#-----------------------------------------------------------------------------
# Running the robot

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=False)

    robot.gfx.addStaticPlotFunction(
        y=('diff', lambda inp: inp.analogInputs[1] - inp.analogInputs[2]))

    robot.behavior = MySM()
    robot.behavior.start(traceTasks = robot.gfx.tasks())

def step():
    inp = io.SensorInput()
    (left, right) = inp.analogInputs[1:3]
    print 'left=', left, 'right=', right, 'diff=', left-right
    robot.behavior.step(inp).execute()

def brainStart():
    robot.gfx.clearPlotData()

def brainStop():
    pass

def shutdown():
    pass
