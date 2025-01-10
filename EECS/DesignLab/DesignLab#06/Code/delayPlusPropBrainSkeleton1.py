import math
import lib601.sm as sm
from soar.io import io
import lib601.gfx as gfx
import lib601.util as util
import lib601.sonarDist as sonarDist

desiredRight = 0.4
forwardVelocity = 0.1
k1 = 10
# k1 = [10, 30, 100, 300]
k2 = -9.97
# k2 = [-9.97, -29.97, -97.35, -271.73]

class Sensor(sm.SM): #input io, output distance to wall
    def getNextValues(self, state, inp):
        v = sonarDist.getDistanceRight(inp.sonars)
        print 'Dist from robot center to wall on right', v
        return (state, v)

# inp is the distance to the right
class WallFollower(sm.SM):
    startState = [desiredRight,'False']
    def getNextValues(self, state, inp):
        e1 = desiredRight - inp
        e2 = desiredRight - state[0]
        w = k1*e1 +k2*e2
        return ([inp,'False'],io.Action(fvel = forwardVelocity,rvel = w))

sensorMachine = Sensor()
sensorMachine.name = 'sensor'
mySM = sm.Cascade(sensorMachine, WallFollower())

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=False)
    robot.gfx.addStaticPlotSMProbe(y=('rightDistance', 'sensor',
                                      'output', lambda x:x))
    robot.behavior = mySM
    robot.behavior.start(traceTasks = robot.gfx.tasks())

def step():
    robot.behavior.step(io.SensorInput()).execute()
    io.done(robot.behavior.isDone())

def brainStop():
    pass
