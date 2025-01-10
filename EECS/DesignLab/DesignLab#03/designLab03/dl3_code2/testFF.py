import lib601.io as io
import lib601.util as util
import math
import ffSkeleton
reload(ffSkeleton)

# Robot is trying to drive in a rotated square
points = [util.Point(0.5, 0.5), util.Point(0.0, 1.0),
          util.Point(-0.5, 0.5), util.Point(0.0, 0.0)]

def testFF():
    sonars = [0.0] * 8
    poseList = [util.Pose(0, 0, 0), util.Pose(0, 1, 0),
                util.Pose(0.499, 0.501, 2), util.Pose(2, 3, 4)]
    ffTestInput = [io.FakeSensorInput(sonars, pose) for pose in poseList]
    targetGenerator = ffSkeleton.FollowFigure(points)
    result = targetGenerator.transduce(ffTestInput, check = True)
    print 'The actual inputs are whole instances of io.SensorInput;  here we'
    print 'are just showing the odometry part of the input.'
    for (i, o) in zip(poseList, result):
        print '\nInput:', i
        print 'Output:', o

