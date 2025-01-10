import lib601.sf as sf
import lib601.sig as sig
import lib601.ts as ts

# 6.01 HomeWork 2 Skeleton File

#Constants relating to some properties of the motor
k_m = 1000
k_b = 0.5
k_s = 5
r_m = 20

def controllerAndSensorModel(k_c):
    pass #your code here

def integrator(T):
    pass #your code here

def motorModel(T):
    pass #your code here

def plantModel(T):
    pass #your code here

def lightTrackerModel(T,k_c):
    pass #your code here


def plotOutput(sfModel):
    """Plot the output of the given SF, with a unit-step signal as input"""
    smModel = sfModel.differenceEquation().stateMachine()
    outSig = ts.TransducedSignal(sig.StepSignal(), smModel)
    outSig.plot()
