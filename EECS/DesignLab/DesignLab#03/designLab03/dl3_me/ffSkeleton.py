# -*- coding: utf-8 -*-
import lib601.sm as sm
import dynamicMoveToPointSkeleton
reload(dynamicMoveToPointSkeleton)

class FollowFigure(sm.SM): # 根据机器人当前的位置决定下一个目标点
    def __init__(self, squarePoints):
        self.squarePoints = squarePoints # 目标点列表，在被调用的时候应该一起传过来
    def getNextValues(self, state, inp):
        if state == 0 or state == None: # 初始位置
            Nowstate = 0
        else:
            Nowstate = state 
        goalPoint = self.squarePoints[Nowstate] # 获取目标点
        if inp.odometry.point().isNear(goalPoint, 0.01) and state < len(self.squarePoints): # 0.01是容差（允许的误差范围），如果当前位置离目标之间的距离小于或等于 0.01，则 isNear() 返回 True；否则返回 False
            nextState = Nowstate + 1 # 判断有没有到点，有没有走完；到点没走完加一，走完停止；没到点保持原状态继续。
        else:
            nextState = Nowstate
        return (nextState, goalPoint)