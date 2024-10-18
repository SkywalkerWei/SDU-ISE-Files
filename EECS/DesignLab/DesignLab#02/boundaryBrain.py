import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io

class MySMClass(sm.SM):
    Startstate = (None, io.SensorInput().sonars[7])
    def getNextValues(self, state, _input):
        nextState = '1'
        forwardSpeed = 0
        rotateSpeed = 0
        if state is None or state[0] == '0':
            aver = (_input.sonars[3]+_input.sonars[4])/2
            if aver > 0.5:
                forwardSpeed = 0.9
                nextState = '0'
            elif aver < 0.35:
                forwardSpeed = -0.3
                nextState = '0'
            else:
                forwardSpeed = 0
                nextState = ('2', _input.sonars[7])
            rotateSpeed = 0
        else:
            if state[0] == '1':   
                if _input.sonars[3] < 0.5 or _input.sonars[4] < 0.5:   
                    nextState = '2'
                    forwardSpeed = 0
                    rotateSpeed = 0
                elif (_input.sonars[7]-state[1])>0.5:
                    nextState = '3'
                    forwardSpeed = 0
                    rotateSpeed = 0
                else: 
                    _side = _input.sonars[6] / _input.sonars[7]
                    lowerLimit = 1.3
                    upperLimit = 1.4
                    sideDistance = _input.sonars[7]
                    distanceToWall = min(_input.sonars[6],_input.sonars[7])
                    if _side < lowerLimit:   
                        if distanceToWall > 0.45: 
                            forwardSpeed = 0.9
                            rotateSpeed = 0   
                        else:   
                            forwardSpeed = 0.3
                            rotateSpeed = 0.9   
                    elif _side > upperLimit: 
                        if distanceToWall > 0.45:  
                            forwardSpeed = 0.3
                            rotateSpeed = -0.9  
                        else:   
                            forwardSpeed = 0.9
                            rotateSpeed = 0
                    else:   
                        if sideDistance > 0.45:
                            forwardSpeed = 0.9
                            rotateSpeed = 0.6  
                        elif sideDistance < 0.35:
                            forwardSpeed = 0.9
                            rotateSpeed = -0.6    
                        else:
                            forwardSpeed = 0.9
                            rotateSpeed = 0    
                    nextState = '1'
            if state[0] == '2':
                if _input.sonars[3] < 0.5 or _input.sonars[4] < 0.5: 
                    forwardSpeed = 0
                    rotateSpeed = 0.9
                    nextState = '2'
                else:       
                    forwardSpeed = 0
                    rotateSpeed = 0
                    nextState = '1'
            if state[0] == '3':
                _side = _input.sonars[6]/_input.sonars[7]
                lowerLimit = 1.3
                upperLimit = 1.4
                if lowerLimit < _side and _side < upperLimit and (_input.sonars[7] < 0.5 or _input.sonars[6] < 0.5): 
                    forwardSpeed = 0
                    rotateSpeed =  0   
                    nextState = '1'
                else:
                    forwardSpeed = 0.3 
                    rotateSpeed = -0.7
                    nextState = '3'
        return ((nextState, _input.sonars[7]), io.Action(fvel = forwardSpeed, rvel = rotateSpeed))

mySM = MySMClass()
mySM.name = 'brainSM'

def plotSonar(sonarNum):
    robot.gfx.addDynamicPlotFunction(y=('sonar'+str(sonarNum), lambda: io.SensorInput().sonars[sonarNum]))

# this function is called when the brain is (re)loaded
def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=True, sonarMonitor=True)
    robot.behavior = mySM

# this function is called when the start button is pushed
def brainStart():
    robot.behavior.start(traceTasks = robot.gfx.tasks())

# this function is called 10 times per second
def step():
    _input = io.SensorInput()
    robot.behavior.step(_input).execute()
    io.done(robot.behavior.isDone())

# called when the stop button is pushed
def brainStop():
    pass

# called when brain or world is reloaded (before setup)
def shutdown():
    pass
