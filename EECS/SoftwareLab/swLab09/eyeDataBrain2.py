import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
import lib601.fr as fr
from soar.io import io

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=False)
    startTheta = io.SensorInput().odometry.theta
    # static plot of robot angle vs eye voltage
    robot.gfx.addStaticPlotFunction(
        x=('angle', lambda inp: util.fixAnglePlusMinusPi(
         (inp.odometry.theta-startTheta))),
        y=('eye voltage', lambda inp: inp.analogInputs[1]))

def brainStart():
    # -1 is clockwise
    dir = 1
    # rotate the robot by (almost) 180 degrees
    robot.behavior = fr.RotateTSM(dir*(math.pi-0.05))
    robot.behavior.start(robot.gfx.tasks())

def step():
    inp = io.SensorInput()
    (neck, left, right) = inp.analogInputs[0:3]
    print 'neck=', neck, 'left=', left, 'right=', right
    robot.behavior.step(inp).execute()

def brainStop():
    pass

def shutdown():
    pass
