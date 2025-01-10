import os
labPath = os.getcwd()
from sys import path
if not labPath in path:
    path.append(labPath)
print 'setting labPath to', labPath

# 这部分的作用：确保当前运行脚本的目录能够被识别为模块搜索路径，从而可以导入同一目录下的其他模块文件

import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io
import dynamicMoveToPointSkeleton
reload(dynamicMoveToPointSkeleton)
import ffSkeleton
reload(ffSkeleton)
from secretMessage import secret

verbose = False
squarePoints = [util.Point(0.5, 0.5), util.Point(0.0, 1.0), util.Point(-0.5, 0.5), util.Point(0.0, 0.0)]
# squarePoints = secret # 使用secretMessage中的路径
mySM = sm.Cascade(sm.Parallel(ffSkeleton.FollowFigure(squarePoints),sm.Wire()),sm.Switch(lambda x: min(x[1].sonars)>0.3,dynamicMoveToPointSkeleton.DynamicMoveToPoint(),sm.Constant(io.Action(fvel=0,rvel=0))))

# 将两个状态机按顺序连接起来，前一个状态机的输出会作为后一个状态机的输入
# 左侧部分：并行运行两个状态机ffSkeleton.FollowFigure(squarePoints)和sm.Wire()，每个状态机独立处理它的输入，产生输出。
# ffSkeleton负责让机器人按顺序移动到预定的目标点，squarePoints 是四个目标点的列表，详情查看相应文件。返回元组（下一状态，目标点）
# sm.Wire 传递的是传感器输入（如机器人传感器数据），供后续状态机使用，这两个要并行，获取当前位置和目标的差异+传感器反馈+判断障碍
# 右侧部分：使用条件状态机Switch，根据传感器输入的值来切换不同的行为逻辑，lambda x: min(x[1].sonars) 用于决定何时切换。
# 传感器读取障碍物距离，min(x[1].sonars) 返回最小值，表示周围最近的障碍物距离。如果距离大那运行 dynamicMoveToPointSkeleton.DynamicMoveToPoint()；否则速度归零。

def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail = True)
    robot.behavior = mySM

def brainStart():
    robot.behavior.start(traceTasks = robot.gfx.tasks(),
                         verbose = verbose)

def step():
    robot.behavior.step(io.SensorInput()).execute()
    io.done(robot.behavior.isDone())

def brainStop():
    pass

def shutdown():
    pass