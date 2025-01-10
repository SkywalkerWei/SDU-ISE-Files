import math
import lib601.sm as sm
from soar.io import io
import lib601.gfx as gfx
import lib601.util as util
import lib601.sonarDist as sonarDist

desiredRight = 0.4
forwardVelocity = 0.1
k3 = 1
# k3 = [1, 3, 10, 30]
k4 = 0.63
# k4 = [0.63, 1.09, 2, 3.46]

# No additional delay.
# Output is a sequence of (distance, angle) pairs
class Sensor(sm.SM):
   def getNextValues(self, state, inp):
       v = sonarDist.getDistanceRightAndAngle(inp.sonars)
       print 'Dist from robot center to wall on right', v[0]
       if not v[1]:
           print '******  Angle reading not valid  ******'
       return (state, v)


# inp is a tuple (distanceRight, angle)
class WallFollower(sm.SM):
    startState = 'False'
    def getNextValues(self, state, inp):
        (distanceRight, angle) = inp
        e1 = desiredRight - distanceRight
        e2 = k4*angle
        w = k3*e1 - k4*e2
        return ('False',io.Action(fvel = forwardVelocity,rvel = w))

sensorMachine = Sensor()
sensorMachine.name = 'sensor'
mySM = sm.Cascade(sensorMachine, WallFollower())

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=False)
    robot.gfx.addStaticPlotSMProbe(y=('rightDistance', 'sensor','output', lambda x:x[0]))
    robot.behavior = mySM
    robot.behavior.start(traceTasks = robot.gfx.tasks())

def step():
    robot.behavior.step(io.SensorInput()).execute()

def brainStop():
    pass
