import lib601.sf as sf
import lib601.sig as sig
import lib601.ts as ts
import lib601.optimize as op

k_m = 1000
k_b = 0.5
k_s = 5
r_m = 20

def controllerAndSensorModel(k_c):
    return sf.Cascade(sf.Gain(k_s), sf.Gain(k_c))

def integrator(T):
    return sf.Cascade(sf.Gain(T),sf.FeedbackAdd(sf.R(), sf.Gain(1)))

def motorModel(T):
    return sf.Cascade(sf.Gain(T*k_m/r_m), sf.FeedbackAdd(sf.R(), sf.Gain(1-k_b*k_m*T/r_m)))

def plantModel(T):
    return sf.Cascade(motorModel(T), integrator(T))

def lightTrackerModel(T,k_c):
    return sf.FeedbackSubtract(sf.Cascade(controllerAndSensorModel(k_c), plantModel(T)), sf.Gain(1))

def plotOutput(sfModel):
    outSig = ts.TransducedSignal(sig.StepSignal(), sfModel.differenceEquation().stateMachine())
    outSig.plot()

def pole(k_c):
    return abs(lightTrackerModel(0.005,k_c).dominantPole())

# print(op.optOverLine(pole,0,100,20000))

# print(op.optOverLine(lambda x: x**2 - x, -20, 20, 10000))

# print(op.optOverLine(lambda x: x**5 - 7*x**3 + 6*x**2 + 2, -1, 2, 10000))

plotOutput(lightTrackerModel(0.005,0.625))