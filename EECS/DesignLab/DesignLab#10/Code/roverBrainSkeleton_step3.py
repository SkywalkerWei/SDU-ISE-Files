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

from boundaryFollower import boundaryFollowerClass

class boundary_light_Follower(boundaryFollowerClass):
	def __init__(self):
		pass
	def getNextValues(self, state, inp):
		v1, v2 = inp.analogInputs[2], inp.analogInputs[1]
		V0 = max(0, min(10, (5 * (v2 - v1) / 2.08) * 0.5 + 5))
		V_light = 6.2
		if v2 > V_light:
			return state, io.Action(fvel = 0.1, rvel = 0.3 * (V0 - 5), voltage=V0)
		else:
			return boundaryFollowerClass.getNextValues(self, state, inp)

mySM = boundary_light_Follower()
mySM.name = 'brainSM'

# mySM = boundaryFollowerClass()

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