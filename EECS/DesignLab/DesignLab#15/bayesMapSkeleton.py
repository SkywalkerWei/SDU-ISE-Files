import lib601.dist as dist
import lib601.util as util
import lib601.colors as colors
import lib601.ssm as ssm
import lib601.seFast as seFast
import lib601.dynamicGridMap as dynamicGridMap


# Define the stochastic state-machine model for a given cell here.

# Observation model:  P(obs | state)
def oGivenS(s):
    if s == 'void':
        return dist.DDist({'hit':0.02,'free':0.98})
    elif s == 'ocp':
        return dist.DDist({'hit':0.95,'free':0.05})
    #elif s == 'unreached':
     #   return dist.DDist({'hit':0.08,'void':0.92})
# Transition model: P(newState | s | a)
def uGivenAS(a): # action is not used
    def transition(s):
        if s == 'void':
            return dist.DDist({'void':1,'ocp':0}) # considering someone walk in
        elif s == 'ocp':
            return dist.DDist({'ocp':1,'void':0})
    return transition
startDist = dist.DDist({'void':0.95,'ocp':0.05}) # initial belief
cellSSM = ssm.StochasticSM(startDist, uGivenAS, oGivenS)   # Your code here

th = 0.8# probability threshold for the 'occupied' function

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
        probDist = self.GridSM[xIndex][yIndex].state
        return probDist.prob('ocp')
    def makeStartingGrid(self):
        def makeSMandStart(x,y): # make a sm and start it
            sm = seFast.StateEstimator(ssm.StochasticSM(startDist, uGivenAS, oGivenS))
            sm.start()
            return sm
        self.GridSM=util.make2DArrayFill(self.xN,self.yN,\
                    makeSMandStart)        
    def setCell(self, (xIndex, yIndex)):
        self.GridSM[xIndex][yIndex].step(('hit',None))
        # redraw square
        self.drawSquare((xIndex,yIndex))
    def clearCell(self, (xIndex, yIndex)):
        self.GridSM[xIndex][yIndex].step(('free',None))
        # redraw square
        self.drawSquare((xIndex,yIndex))     
    def occupied(self, (xIndex, yIndex)):
        return self.occProb((xIndex,yIndex)) > th        

mostlyHits = [('hit', None), ('hit', None), ('hit', None), ('free', None)]
mostlyFree = [('free', None), ('free', None), ('free', None), ('hit', None)]

def testCellDynamics(cellSSM, input):
    se = seFast.StateEstimator(cellSSM)
    return se.transduce(input)

