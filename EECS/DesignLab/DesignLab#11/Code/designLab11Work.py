import lib601.dist as dist
import lib601.coloredHall as coloredHall
from lib601.coloredHall import *

standardHallway = ['white', 'white', 'green', 'white', 'white']
alternating = ['white', 'green'] * 6
sterile = ['white'] * 16
testHallway = ['chocolate', 'white', 'green', 'white', 'white',
               'green', 'green', 'white',  
               'green', 'white', 'green', 'chocolate']

maxAction = 5
actions = [str(x) for x in range(maxAction) + [-x for x in range(1, maxAction)]]

def makePerfect(hallway = standardHallway):
    return makeSim(hallway, actions, perfectObsNoiseModel,
                   standardDynamics, perfectTransNoiseModel,'perfect')
def makeNoisy(hallway = standardHallway):
    return  makeSim(hallway, actions, noisyObsNoiseModel, standardDynamics,
                    noisyTransNoiseModel, 'noisy')
n=makeNoisy()
n.run(20)
def makeNoisyKnownInitLoc(initLoc, hallway = standardHallway):
    return  makeSim(hallway, actions, noisyObsNoiseModel, standardDynamics,
                    noisyTransNoiseModel, 'known init',
                    initialDist = dist.DDist({initLoc: 1}))



