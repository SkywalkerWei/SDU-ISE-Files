import lib601.dist as dist
import lib601.util as util
import lib601.colors as colors
import lib601.ssm as ssm
import lib601.seFast as seFast
import lib601.dynamicGridMap as dynamicGridMap


# Define the stochastic state-machine model for a given cell here.

# Observation model:  P(obs | state)
def oGivenS(s):
    pass    
# Transition model: P(newState | s | a)
def uGivenAS(a):
     pass    
cellSSM = None   # Your code here

class BayesGridMap(dynamicGridMap.DynamicGridMap):

    def squareColor(self, (xIndex, yIndex)):
        p = self.occProb((xIndex, yIndex))
        if self.robotCanOccupy((xIndex,yIndex)):
            return colors.probToMapColor(p, colors.greenHue)
        elif self.occupied((xIndex, yIndex)):
            return 'black'
        else:
            return 'red'
        
    def occProb(self, (xIndex, yIndex)):
        pass        
    def makeStartingGrid(self):
        pass        
    def setCell(self, (xIndex, yIndex)):
        pass        
    def clearCell(self, (xIndex, yIndex)):
        pass        
    def occupied(self, (xIndex, yIndex)):
        pass        

mostlyHits = [('hit', None), ('hit', None), ('hit', None), ('free', None)]
mostlyFree = [('free', None), ('free', None), ('free', None), ('hit', None)]

def testCellDynamics(cellSSM, input):
    se = seFast.StateEstimator(cellSSM)
    return se.transduce(input)

