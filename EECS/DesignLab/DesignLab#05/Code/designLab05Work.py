import lib601.sig as sig
import lib601.ts as ts
import lib601.poly as poly
import lib601.sf as sf

import matplotlib.pyplot as plt

def controller(k):
   return sf.Gain(k)

def plant1(T):
   return sf.SystemFunction(poly.Polynomial([T, 0]), poly.Polynomial([-1, 1]))
   # return sf.Cascade(sf.Gain(T),sf.FeedbackAdd(sf.R()))

def plant2(T, V):
   # return sf.SystemFunction(poly.Polynomial([V*T, 0]), poly.Polynomial([-1, 1]))
   return sf.Cascade(sf.Gain(V*T),sf.FeedbackAdd(sf.R()))

def wallFollowerModel(k, T, V):
   return sf.FeedbackSubtract(sf.Cascade(sf.Cascade(controller(k), plant1(T)), plant2(T, V)), sf.Gain(1))

gains = [0.1, 0.5, 1.0, 2.0]

for k in gains:
   # print(f"Gain: {k}, Dominant Pole: {wallFollowerModel(k, 0.1, 0.1).dominantPole()}")
   print "Gain: %f, Dominant Pole: %s" % (k, wallFollowerModel(k, 0.1, 0.1).dominantPole())