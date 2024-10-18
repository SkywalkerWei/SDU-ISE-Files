import math
import lib601.util as util
import lib601.sm as sm
import lib601.gfx as gfx
from soar.io import io

class MySMClass(sm.SM):
    def getNextValues(self, state, inp):
        # 目标距离（保持与障碍物的距离）
        TargetDistance = 0.5

        # 当前距离
        currentDistance = ( inp.sonars[3] + inp.sonars[4] )/2

        # 当前误差
        error = currentDistance - TargetDistance

        # 比例增益（Kp 值，可以通过实验调试找到合适的值）
        Kp = 0.5

        # 计算前进速度，使用比例控制器
        fvel = Kp * error

        rvel = 0
        # 限制前进速度在 -0.3 到 0.3 之间
        if fvel > 0.1:
            fvel = 0.1
        elif fvel < -0.1:
            fvel = -0.1

        # 返回状态（保持不变）和行动
        return (state, io.Action(fvel, rvel))

mySM = MySMClass()
mySM.name = 'brainSM'

######################################################################
###
###          Brain methods
###
######################################################################

def plotSonar(sonarNum):
    robot.gfx.addDynamicPlotFunction(y=('sonar'+str(sonarNum),
                                        lambda: 
                                        io.SensorInput().sonars[sonarNum]))

# this function is called when the brain is (re)loaded
def setup():
    robot.gfx = gfx.RobotGraphics(drawSlimeTrail=True, # slime trails
                                  sonarMonitor=True) # sonar monitor widget
    
    # set robot's behavior
    robot.behavior = mySM

# this function is called when the start button is pushed
def brainStart():
    robot.behavior.start(traceTasks = robot.gfx.tasks())

# this function is called 10 times per second
# 这是代码的核心执行逻辑，每秒被调用 10 次。
# io.SensorInput(): 获取机器人当前的传感器输入。
# robot.behavior.step(inp).execute(): 使用传感器输入调用状态机的 step 方法，计算并执行相应的动作。
def step():
    inp = io.SensorInput()
    print inp.sonars[0]
    print inp.sonars[1]
    robot.behavior.step(inp).execute()
    io.done(robot.behavior.isDone())

# called when the stop button is pushed
def brainStop():
    pass

# called when brain or world is reloaded (before setup)
def shutdown():
    pass
