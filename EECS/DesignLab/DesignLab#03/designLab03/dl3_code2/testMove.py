import lib601.io as io
import lib601.util as util
import math
import dynamicMoveToPointSkeleton
reload(dynamicMoveToPointSkeleton)

def testMove():
    target = util.Point(1.0, 0.5)
    sonars = [0.0] * 8
    poseList = [util.Pose(0, 0, 0),
                util.Pose(0, 0, math.pi/2),
                util.Pose(0, 0, math.atan2(0.5, 1)),
                util.Pose(1.0001, 0.499999, 0)]
    moveTestInput = [(target, io.FakeSensorInput(sonars, pose)) \
                                          for pose in poseList]
    mover = dynamicMoveToPointSkeleton.DynamicMoveToPoint()
    result = mover.transduce(moveTestInput, check = True)
    print 'The actual inputs are (target, sensorInput) pairs;  here we'
    print 'are just showing the odometry part of the sensorInput.'
    for ((goal, sensors), o) in zip(moveTestInput, result):
        print '\nInput:', (goal, sensors.odometry)
        print 'Output:', o
    
                            
    

