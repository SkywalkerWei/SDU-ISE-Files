import math
import lib601.sig as sig
import cmax.simulate as simulate

def testSignal(dist = 3.0, simTime = 3.0):
   nsteps = int(simTime/simulate.Tsim)
   print __name__, 'nsteps ', nsteps
   ninter=nsteps/4
   return (nsteps,
           sig.ListSignal(ninter*[{'lightAngle':1.57, 'lightDist':dist}]+\
                          ninter*[{'lightAngle':3.14, 'lightDist':dist}]+\
                          ninter*[{'lightAngle':1.57, 'lightDist':dist}]+\
                          ninter*[{'lightAngle':3.14, 'lightDist':dist}]))

(nsteps, sigIn) = testSignal(dist=3.0)
def runTest(lines, parent = None, nsteps = nsteps, meter=None, display=True):
   return simulate.runCircuit(lines, sigIn, parent, nsteps, meter, display)

def simpleSignal(dist = 3.0, simTime = 3.0):
   nsteps = int(simTime/simulate.Tsim)
   return (nsteps,
           sig.ListSignal(nsteps*[{'lightAngle':1.57, 'lightDist':dist}]))

def simpleSignal2(dist = 3.0, simTime = 3.0):
   nsteps = int(simTime/simulate.Tsim)
   return (nsteps,
           sig.ListSignal(nsteps*[{'lightAngle':3*math.pi/4, 'lightDist':dist}]))
