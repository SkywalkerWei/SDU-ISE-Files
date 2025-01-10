import lib601.util as util
import lib601.dist as dist
import lib601.distPlot as distPlot
import lib601.sm as sm
import lib601.ssm as ssm
import lib601.sonarDist as sonarDist
import lib601.move as move
import lib601.seGraphics as seGraphics
import lib601.idealReadings as idealReadings

# For testing your preprocessor
class SensorInput:
    def __init__(self, sonars, odometry):
        self.sonars = sonars
        self.odometry = odometry

preProcessTestData = [SensorInput([0.8, 1.0], util.Pose(1.0, 0.5, 0.0)),
                       SensorInput([0.25, 1.2], util.Pose(2.4, 0.5, 0.0)),
                       SensorInput([0.16, 0.2], util.Pose(7.3, 0.5, 0.0))]
testIdealReadings = ( 5, 1, 1, 5, 1, 1, 1, 5, 1, 5 )
testIdealReadings100 = ( 50, 10, 10, 50, 10, 10, 10, 50, 10, 50 )


class PreProcess(sm.SM):
    
    def __init__(self, numObservations, stateWidth):
        self.startState = (None, None)
        self.numObservations = numObservations
        self.stateWidth = stateWidth

    def getNextValues(self, state, inp):
        (lastUpdatePose, lastUpdateSonar) = state
        currentPose = inp.odometry
        currentSonar = idealReadings.discreteSonar(inp.sonars[0],
                                                   self.numObservations)
        # Handle the first step
        if lastUpdatePose == None:
            return ((currentPose, currentSonar), None)
        else:
            action = discreteAction(lastUpdatePose, currentPose,
                                    self.stateWidth)
            print (lastUpdateSonar, action)
            return ((currentPose, currentSonar), (lastUpdateSonar, action))

# Only works when headed to the right
def discreteAction(oldPose, newPose, stateWidth):
    return int(round(oldPose.distance(newPose) / stateWidth))

def makeRobotNavModel(ideal, xMin, xMax, numStates, numObservations):
    
    startDistribution = dist.squareDist(0,numStates)    # redefine this

    def observationModel(ix):
        noise = dist.squareDist(0, numObservations)
        peak = dist.triangleDist(ideal[ix], round(numObservations / 25.0))
        max_obs = dist.DeltaDist(numObservations)
        return dist.MixtureDist(dist.MixtureDist(peak, noise, 0.9), max_obs, 0.88)


    def transitionModel(a):
        # a is a discrete action
        # returns a conditional probability distribution on the next state
        # given the previous state
        def tm(ix):
            peak = ix + a
            hwidth = round(0.04*numStates)
            if peak > (numStates - 1):
                peak = numStates - 1
            return dist.triangleDist(peak, hwidth, lo = 0, hi = numStates - 1)
        return tm

    return ssm.StochasticSM(startDistribution, transitionModel,
                            observationModel)
# Main procedure
def makeLineLocalizer(numObservations, numStates, ideal, xMin, xMax, robotY):
    Driver = move.MoveToFixedPose(util.Pose(xMax, robotY, 0.0),maxVel = 0.5)
    mySM = makeRobotNavModel(ideal, xMin, xMax, numStates, numObservations)
    ppEst = sm.Cascade(PreProcess(numObservations, (xMax - xMin) / numStates),seGraphics.StateEstimator(mySM))
    return sm.Cascade(sm.Parallel(ppEst, Driver), sm.Select(1))

ppEst = sm.Cascade(PreProcess(10,1.0),seGraphics.StateEstimator(makeRobotNavModel(testIdealReadings,0.0,10.0,10,10)))
ppEst.transduce(preProcessTestData)
