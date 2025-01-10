import lib601.sig  as sig
import lib601.ts as ts
import lib601.sm as sm

def plant(time_step, initial_distance):
    """
    模拟机器人与障碍物之间的距离变化
    """
    return sm.SM(startState=initial_distance, getNextValues=lambda state, input_value: (state - input_value * time_step, state - input_value * time_step))

def controller(gain):
    """
    根据距离误差计算机器人的控制输出
    """
    return sm.PureFunction(lambda distance_e: -gain * distance_e)

def sensor(initial_distance):
    """
    模拟距离传感器，返回延迟后的距离测量值
    """
    return sm.Delay(initial_distance)

def wall_finder_system(time_step, initial_distance, gain):
    """
    整合控制器和传感器，构建完整的墙壁识别系统
    """
    return sm.FeedbackSubtract(sm.Cascade(controller(gain), plant(time_step, initial_distance)), sensor(initial_distance))

INITIAL_DISTANCE = 1.5

def plot_wall_finder(gain, end_time):
    """
    使用不同的增益系数绘制墙壁识别系统的响应曲线
    """
    ts.TransducedSignal(sig.ConstantSignal(0.7), wall_finder_system(0.1, INITIAL_DISTANCE, gain)).plot(0, end_time, newWindow=f'Gain {gain}')

plot_wall_finder(18, 50)