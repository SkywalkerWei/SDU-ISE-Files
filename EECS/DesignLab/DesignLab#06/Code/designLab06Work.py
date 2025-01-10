import lib601.sf as sf
import lib601.optimize as optimize
import operator

def delayPlusPropModel(k1, k2):
    T = 0.1
    V = 0.1
    
    # Controller:  your code here
    controller = None
    # The plant is like the one for the proportional controller.  Use
    # your definition from last week.
    plant1 = None
    plant2 = None
    # Combine the three parts
    sys = None
    return sys

# You might want to define, and then use this function to find a good
# value for k2.

# Given k1, return the value of k2 for which the system converges most
# quickly, within the range k2Min, k2Max.  Should call optimize.optOverLine.

def bestk2(k1, k2Min, k2Max, numSteps):
    pass


def anglePlusPropModel(k3, k4):
    T = 0.1
    V = 0.1

    # plant 1 is as before
    plant1 = None
    # plant2 is as before
    plant2 = None
    # The complete system
    sys = None
    
    return sys


# Given k3, return the value of k4 for which the system converges most
# quickly, within the range k4Min, k4Max.  Should call optimize.optOverLine.

def bestk4(k3, k4Min, k4Max, numSteps):
    pass
