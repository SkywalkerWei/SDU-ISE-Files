"""
Drive robot to goal specified as odometry pose.
"""
from soar.io import io
import lib601.sm as sm
import lib601.util as util

class MoveToDynamicPoint(sm.SM):
    """
    Drive to a goal point in the frame defined by the odometry.  Goal
    points are part of the input, in contrast to C{MoveToFixedPoint},
    which takes a single goal point at initialization time.

    Assume inputs are C{(util.Point, io.SensorInput)} pairs

    This is really a pure function machine;  defining its own class,
    though, so we can easily modify the parameters.
    """
    forwardGain = 1.0
    """Gain for driving forward"""
    rotationGain = 0.5
    """Gain for rotating"""
    maxVel = 0.5
    """Maximum velocity"""
    angleEps = 0.1
    """Tolerance for angles"""

    def getNextValues(self, state, inp):
        (goalPoint, sensors) = inp
        return (None, actionToPoint(goalPoint, sensors.odometry,
                                    self.forwardGain, self.rotationGain,
                                    self.maxVel, self.angleEps))

def actionToPoint(goalPoint, robotPose, forwardGain, rotationGain,
                  maxVel, angleEps):
    """
    Internal procedure that returns an action to take to drive
    toward a specified goal point.
    """
    rvel = 0
    fvel = 0
    robotPoint = robotPose.point()
    
    # Compute the angle between the robot and the goal point
    headingTheta = robotPoint.angleTo(goalPoint)
    
    # Difference between the way the robot is currently heading and
    # the angle to the goal.  This is an angular error relative to the
    # robot's current heading, in the range +pi to -pi.
    headingError = util.fixAnglePlusMinusPi(headingTheta - robotPose.theta)
    
    if util.nearAngle(robotPose.theta, headingTheta, angleEps):
        # The robot is pointing toward the goal, so it's okay to move
        # forward.  Note that we are actually doing two proportional
        # controllers simultaneously:  one to reduce angular error
        # and one to reduce distance error.
        distToGoal = robotPoint.distance(goalPoint)
        fvel = distToGoal * forwardGain
        rvel = headingError * rotationGain
    else:
        # The robot is not pointing close enough to the goal, so don't
        # start moving foward yet.  This is a proportional controller
        # to reduce angular error.
        rvel = headingError * rotationGain
    return io.Action(fvel = util.clip(fvel, -maxVel, maxVel),
                     rvel = util.clip(rvel, -maxVel, maxVel))



class MoveToFixedPose(sm.SM):
    """
    State machine representing robot behavior that drives to a
    specified pose.  Inputs are instances of C{io.SensorInput};
    outputs are instances of C{io.Action}.   Robot first rotates
    toward goal, then moves straight, then rotates to desired final
    angle. 
    """
    forwardGain = 1.0
    """Gain for driving forward"""
    rotationGain = 1.0
    """Gain for rotating"""
    maxVel = 0.5
    """Maximum velocity"""
    angleEps = 0.05
    """Tolerance for angles"""
    distEps = 0.05
    """Tolerance for distances"""

    startState = False
  
    def __init__(self, goalPose, maxVel = maxVel):
        """
        @param goalPose: instance of C{util.Pose} specifying goal for
        robot in odometry coordinates
        """
        self.goalPose = goalPose
        self.maxVel = maxVel

    def getNextValues(self, state, inp):
        nearGoal = inp.odometry.near(self.goalPose, self.distEps, self.angleEps)
        return (nearGoal, actionToPose(self.goalPose, inp.odometry,
                                        self.forwardGain, self.rotationGain,
                                        self.maxVel, self.angleEps,
                                       self.distEps))

    def done(self, state):
        return state

class MoveToFixedPoint(sm.SM):
    """
    State machine representing robot behavior that drives to a
    specified point.  Inputs are instances of C{io.SensorInput};
    outputs are instances of C{io.Action}.   Robot first rotates
    toward goal, then moves straight.  It will correct its rotation if
    necessary.  
    """
    forwardGain = 1.0
    """Gain for driving forward"""
    rotationGain = 1.0
    """Gain for rotating"""
    angleEps = 0.05
    """Tolerance for angles"""
    distEps = 0.05
    """Tolerance for distances"""
    maxVel = 0.5
    """Maximum velocity"""
    
    startState = False
  
    def __init__(self, goalPoint, maxVel = maxVel):
        self.goalPoint = goalPoint
        self.maxVel = maxVel

    def getNextValues(self, state, inp):
        nearGoal = inp.odometry.point().isNear(self.goalPoint, self.distEps)
        return (nearGoal, actionToPoint(self.goalPoint, inp.odometry,
                                        self.forwardGain, self.rotationGain,
                                        self.maxVel, self.angleEps))

    def done(self, state):
        return state


def actionToPose(goalPose, robotPose, forwardGain, rotationGain,
                 maxVel, angleEps, distEps):
    """
    Internal procedure that returns an action to take to drive
    toward a specified goal pose.
    """
    if robotPose.distance(goalPose) < distEps:
        finalRotError = util.fixAnglePlusMinusPi(goalPose.theta -robotPose.theta)
        return io.Action(rvel = finalRotError * rotationGain)
    else:
        return actionToPoint(goalPose.point(), robotPose, forwardGain,
                             rotationGain, maxVel, angleEps)

