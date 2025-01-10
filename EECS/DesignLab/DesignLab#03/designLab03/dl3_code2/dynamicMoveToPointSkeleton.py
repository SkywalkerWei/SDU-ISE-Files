# -*- coding: utf-8 -*-
import lib601.sm as sm
import lib601.util as util
import math

# Use this line for running in idle
# import lib601.io as io
# Use this line for testing in soar
from soar.io import io


class DynamicMoveToPoint(sm.SM):
    def getNextValues(self, state, inp):
        (goal, Sensorinput) = inp
        key_1 = 1.5 # 前进速度比例系数
        key_2 = 2.0 # 旋转速度比例系数
        angle_error = 0.001
        position_error = 0.005
        p1 = Sensorinput.odometry  # 获取传感器输入中的里程计信息 p1，它表示机器人的当前位置和朝向。
    

        inp = (goal, p1)
        distance = goal.distance(p1.point()) #目标点与当前位置的距离
        angle1 = p1.point().angleTo(goal) #当前位置指向目标点的角度
        angle2 = p1.theta #机器人的当前朝向
        go_speed = key_1 * distance
        rot_speed = key_2 * util.fixAnglePlusMinusPi(angle1 - angle2) #旋转速度，根据角度差计算，并使用 util.fixAnglePlusMinusPi 函数将角度差限制在 -π 到 π 之间


        assert isinstance(inp, tuple), 'inp should be a tuple'
        assert len(inp) == 2, 'inp should be of length 2'
        assert isinstance(inp[0], util.Point), 'inp[0] should be a Point'


        if distance > position_error:
            if not (util.nearAngle(angle1, angle2, angle_error)):
                return (state, io.Action(fvel=0, rvel=rot_speed))
            else:
                return (state, io.Action(fvel=go_speed, rvel=0))
        if distance < position_error:
            return (state, io.Action())
        
        # print('iooutput:', io.SensorInput(cheat = True).odometry.point())
        # Replace this definition
        # print 'DynamicMoveToPoint', 'state=', state, 'inp=', inp
