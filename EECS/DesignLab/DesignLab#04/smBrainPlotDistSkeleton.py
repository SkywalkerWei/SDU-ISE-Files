import math
import lib601.sm as sm
from soar.io import io
import lib601.gfx as gfx
import lib601.util as util

DESIRED_DISTANCE = 0.7 # 期望距离

class DistanceController(sm.SM):
    """
    控制机器人的速度以保持期望的距离
    """
    def getNextValues(self, state, sensor_input):
        return (state, io.Action(rvel=0, fvel=2*(-DESIRED_DISTANCE + sensor_input)))

class DistanceSensor(sm.SM):
    """
    传感器模拟器，用于获取距离信息
    """
    def __init__(self, initial_distance, num_delays):
        self.start_state = [initial_distance] * num_delays

    def getNextValues(self, state, _input):
        return ([_input.sonars[3]] + state[:-1], state[-1])

brain_sm = sm.Cascade(DistanceSensor(1.5, 1), DistanceController()) 
brain_sm.name = 'brainSM'

def plot_sonar(sonar_num):
    robot.gfx.addStaticPlotFunction(y=(f'sonar{sonar_num}', lambda: io.Sensor_inputut().sonars[sonar_num]))

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=False)
    plot_sonar(3)
    robot.behavior = brain_sm
    robot.behavior.start(traceTasks=robot.gfx.tasks())

def brain_start():
    pass

def step():
    robot.behavior.step(io.Sensor_inputut()).execute()

def brain_stop():
    pass

def shutdown():
    pass