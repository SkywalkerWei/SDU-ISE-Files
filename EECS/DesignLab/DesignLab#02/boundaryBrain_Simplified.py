import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io

class MySMClass(sm.SM):
    Startstate = (None, io.SensorInput().sonars[7])
    def getNextValues(self, state, _input):
        nextState, forwardSpeed, rotateSpeed = '1', 0, 0
        sonars = _input.sonars
        frontAvg = (sonars[3] + sonars[4]) / 2
        if state is None or state[0] == '0':
            if frontAvg > 0.5:
                forwardSpeed, nextState = 0.9, '0'
            elif frontAvg < 0.35:
                forwardSpeed, nextState = -0.3, '0'
            else:
                nextState = ('2', sonars[7])
        elif state[0] == '1':
            if sonars[3] < 0.5 or sonars[4] < 0.5:
                nextState = '2'
            elif sonars[7] - state[1] > 0.5:
                nextState = '3'
            else:
                _side = sonars[6] / sonars[7]
                distanceToWall = min(sonars[6], sonars[7])
                if _side < 1.3:
                    forwardSpeed = 0.9 if distanceToWall > 0.45 else 0.3
                    rotateSpeed = 0 if distanceToWall > 0.45 else 0.9
                elif _side > 1.4:
                    forwardSpeed = 0.3 if distanceToWall > 0.45 else 0.9
                    rotateSpeed = -0.9 if distanceToWall > 0.45 else 0
                else:
                    if sonars[7] > 0.5:
                        forwardSpeed, rotateSpeed = 0.9, 0.6
                    elif sonars[7] < 0.35:
                        forwardSpeed, rotateSpeed = 0.9, -0.6
                    else:
                        forwardSpeed, rotateSpeed = 0.9, 0
                nextState = '1'
        elif state[0] == '2':
            if sonars[3] < 0.5 or sonars[4] < 0.5:
                nextState, rotateSpeed = '2', 0.9
            else:
                nextState = '1'
        elif state[0] == '3':
            _side = sonars[6] / sonars[7]
            if 1.3 < _side < 1.4 and (sonars[6] < 0.5 or sonars[7] < 0.5):
                nextState = '1'
            else:
                nextState, forwardSpeed, rotateSpeed = '3', 0.3, -0.7
        return ((nextState, sonars[7]), io.Action(fvel=forwardSpeed, rvel=rotateSpeed))

mySM = MySMClass()
mySM.name = 'brainSM'

def plotSonar(sonarNum):
    robot.gfx.addDynamicPlotFunction(y=('sonar' + str(sonarNum), lambda: io.SensorInput().sonars[sonarNum]))

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=True, sonarMonitor=True)
    robot.behavior = mySM

def brainStart():
    robot.behavior.start(traceTasks=robot.gfx.tasks())

def step():
    robot.behavior.step(io.SensorInput()).execute()
    io.done(robot.behavior.isDone())

def brainStop():
    pass

def shutdown():
    pass
