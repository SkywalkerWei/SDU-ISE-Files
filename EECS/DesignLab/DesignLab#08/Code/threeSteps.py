import lib601.sig as sig
import cmax.simulate as simulate

def testSignal(simTime = 2.5):
    nsteps = int(simTime/simulate.Tsim)
    print __name__, 'nsteps ', nsteps
    ninter = nsteps/3
    return (nsteps,
	    sig.ListSignal(ninter*[{'pot1':.25}]+\
                           ninter*[{'pot1':.5}]+\
                           ninter*[{'pot1':.75}]))

(nsteps, sigIn) = testSignal()
def runTest(lines, parent = None, nsteps = nsteps, meter=None, display=True):
    return simulate.runCircuit(lines, sigIn, parent, nsteps, meter, display)

