#!/usr/bin/env python
# encoding: utf-8
# 访问 http://tool.lu/pyc/ 查看更多信息
import lib601.sm as sm
from soar.io import io
robotWidth = 0.2
forwardSpeed = 0.2
rotationalSpeed = 0.2
stop = io.Action(0, 0,voltage=5)
go = io.Action(forwardSpeed, 0,voltage=5)
goFast = io.Action(2 * forwardSpeed, 0,voltage=5)
left = io.Action(0, rotationalSpeed,voltage=5)
right = io.Action(0.3 * rotationalSpeed, - rotationalSpeed,voltage=5)

def clearTest(selectedSensors, threshold):
    return min(selectedSensors) > threshold


def frontClear(sensors):
    return clearTest(frontSonars(sensors), clearDist)


def frontClearFar(sensors):
    return clearTest(frontSonars(sensors), clearFarDist)


def leftClear(sensors):
    return clearTest(leftSonars(sensors), sideClearDist)


def rightClear(sensors):
    return clearTest(rightSonars(sensors), sideClearDist)


def frontSonars(sensors):
    return sensors.sonars[2:6]


def front6Sonars(sensors):
    return sensors.sonars[1:7]


def leftSonars(sensors):
    return sensors.sonars[0:3]


def rightSonars(sensors):
    return sensors.sonars[5:8]


def rightmostSonar(sensors):
    return sensors.sonars[7:8]

sideClearDist = 0.3
clearDist = 0.25
clearFarDist = 0.7

def wallInFront(sensors):
    return not clearTest(front6Sonars(sensors), clearDist)


def wallOnRight(sensors):
    return not clearTest(rightmostSonar(sensors), sideClearDist)


def pickAction(state):
    if state == 'turningLeft':
        return left
    if state == 'turningRight':
        return right
    if state == 'stop':
        return stop
    return go


class boundaryFollowerClass(sm.SM):
    startState = 'movingForward'

    def getNextValues(self, state, inp):
        if state == 'turningLeft':
            if wallInFront(inp):
                nextState = 'turningLeft'
            elif wallOnRight(inp):
                nextState = 'following'
            else:
                nextState = 'turningLeft'
        elif state == 'turningRight':
            if wallOnRight(inp):
                nextState = 'following'
            else:
                nextState = 'turningRight'
        elif state == 'movingForward':
            if wallInFront(inp):
                nextState = 'turningLeft'
            else:
                nextState = 'movingForward'
        elif wallInFront(inp):
            nextState = 'turningLeft'
        elif wallOnRight(inp):
            nextState = 'following'
        else:
            nextState = 'turningRight'
        return (nextState, pickAction(nextState))
