# -*- coding: utf-8 -*-
import lib601.sm as sm
import lib601.util as util
import math
from soar.io import io

class DynamicMoveToPoint(sm.SM):
    def getNextValues(self, state, inp):
        fv, rv = 0, 0
        (goalPoint, sensors) = inp
        sensors = sensors.odometry
        currentPoint = sensors.point()
        currentAngle = util.fixAnglePlusMinusPi(sensors.theta)
        goalAngle = util.fixAnglePlusMinusPi(currentPoint.angleTo(goalPoint))
        rotation = goalAngle-currentAngle    
        moving = currentPoint.distance(goalPoint) # 拿到欧氏距离；moving和rotation用于平滑移动，离得远快，离得近慢
        if not util.nearAngle(currentAngle, goalAngle, 0.01): # m没转到，继续转
            fv, rv = 0, 0.7*rotation + rotation/abs(rotation)*0.05 # 为什么+0.05：否则在离得很近但是没到的时候会走的极其慢，这个相当于最小速度；取绝对值相除是拿到符号，sign()在numpy库，就不开了。
        elif not currentPoint.isNear(goalPoint, 0.01): # 没走到，继续走
            fv, rv = 0.7*moving + moving/abs(moving)*0.05, 0
        else: # 到了，停止
            fv, rv = 0, 0
        return (state, io.Action(fvel = fv, rvel = rv))