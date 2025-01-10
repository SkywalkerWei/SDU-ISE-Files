import lib601.sig as sig
import cmax.simulate as simulate

def testSignal(simTime = 1.0):
    nsteps = int(simTime/simulate.Tsim)
    print __name__, 'nsteps ', nsteps
    ninter = nsteps/2
    return (nsteps,
	    sig.ListSignal(ninter*[{'pot1':0.0}]+\
                           ninter*[{'pot1':0.1}]))

(nsteps, sigIn) = testSignal()
def runTest(lines, parent = None, nsteps = nsteps, meter=None, display=True):
    return simulate.runCircuit(lines, sigIn, parent, nsteps, meter, display)

