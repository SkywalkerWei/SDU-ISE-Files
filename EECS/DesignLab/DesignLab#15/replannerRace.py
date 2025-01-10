"""
State machine classes for planning paths in a grid map.
"""
import lib601.util as util
import lib601.sm as sm
import math
import lib601.ucSearch as ucSearch
import lib601.gridDynamics as gridDynamics
reload(gridDynamics)
##___________________mycodehere____________________________
def simplifySubgoals(subgoalList, currentIndices):
    remainGoals = range(len(subgoalList))
    if len(subgoalList) < 2:
        return subgoalList
    (cx, cy) = currentIndices
    xincre = None
    yincre = None
    for i in range(len(subgoalList)):
        (nx, ny)=subgoalList[i]
        if (nx-cx == xincre) and (ny-cy == yincre): # in a line
            remainGoals.remove(i-1) # delete previous subgoal
        else:
            xincre = nx-cx
            yincre = ny-cy
            (cx,cy)=(nx,ny)
    res = []
    for i in remainGoals:
        res.append(subgoalList[i])
    return res

##class GridDynamics(sm.SM):
##    
##    legalInputs = range(16) # 0 is the upper grid ,in the clockwise sequence
##    # we also add the four further grids
##    def __init__(self, theMap):
##        self.Map = theMap
##
##    def getNextValues(self, state, inp):
##        (ix, iy) = state
##        movingDict = {0:(ix,iy+1),1:(ix+1,iy+1),2:(ix+1,iy),
##                      3:(ix+1,iy-1),4:(ix,iy-1),5:(ix-1,iy-1),
##                      6:(ix-1,iy),7:(ix-1,iy+1),8:(ix+1,iy+2),
##                      9:(ix+2,iy+1),10:(ix+2,iy-1),11:(ix+1,iy-2),
##                      12:(ix-1,iy-2),13:(ix-2,iy-1),14:(ix-2,iy+1),
##                      15:(ix-1,iy+2)}
##        target = movingDict[inp]
##        cost = math.sqrt((ix-target[0])**2 + (iy-target[1])**2)*self.Map.xStep
##        if not self.Map.robotCanOccupy(target):
##            return (state, cost)
##        if inp%2 != 0 and inp <= 7: # move diagonally
##            if (not self.Map.robotCanOccupy(movingDict[(inp+7)%8])) or \
##               (not self.Map.robotCanOccupy(movingDict[(inp+1)%8])):
##                return (state, cost)
##        elif inp > 7:
##            traverseIndices = util.lineIndicesConservative((ix,iy),movingDict[inp])
##            for indice in traverseIndices:
##                if not self.Map.robotCanOccupy(indice):
##                    return (state, cost)
##        return (target, cost)
##___________________________________________________________

class ReplannerWithDynamicMap(sm.SM):
    """
    This replanner state machine has a dynamic map, which is an input
    to the state machine.  Input to the machine is a pair C{(map,
    sensors)}, where C{map} is an instance of a subclass of
    C{gridMap.GridMap} and C{sensors} is an instance of
    C{io.SensorInput};  output is an instance of C{util.Point},
    representing the desired next subgoal.  The planner should
    guarantee that a straight-line path from the current pose to the
    output pose is collision-free in the current map.
    """
    def __init__(self, goalPoint):
        """
        @param goalPoint: fixed goal that the planner keeps trying to
        reach
        """
        self.goalPoint = goalPoint
        self.startState = None
        """
        State is the plan currently being executed.  No plan to start with.
        """

    def getNextValues(self, state, inp):
        (map, sensors) = inp
        # Make a model for planning in this particular map
        dynamicsModel = gridDynamics.GridDynamics(map)
        # Find the indices for the robot's current location and goal
        currentIndices = map.pointToIndices(sensors.odometry.point())
        goalIndices = map.pointToIndices(self.goalPoint)
        
        if timeToReplan(state, currentIndices, map, goalIndices):
            # Define heuristic to be Euclidean distance
            def h(s):
                return self.goalPoint.distance(map.indicesToPoint(s))
            # Define goal test
            def g(s):
                return s == goalIndices
            # Make a new plan
            plan = ucSearch.smSearch(dynamicsModel, currentIndices, g,
                                     heuristic = h, maxNodes = 5000)
            # Clear the old path from the map
            if state: map.undrawPath(state)

            if plan:
                # The call to the planner succeeded;  extract the list
                # of subgoals
                state = [s[:2] for (a, s) in plan]
                print 'New plan', state
                # Draw the plan
                map.drawPath(state)
                
                ##
                state = simplifySubgoals(state,currentIndices)## i add
                ##
                
            else:
                # The call to the plan failed
                # Just show the start and goal indices, for debugging
                map.drawPath([currentIndices, goalIndices])
                state = None
        
        if not state or (currentIndices == state[0] and len(state) == 1):
            # If we don't have a plan or we've already arrived at the
            # goal, just ask the move machine to stay at the current pose.
            return (state, sensors.odometry)
        elif currentIndices == state[0] and len(state) > 1:
            # We have arrived at the next subgoal in our plan;  so we
            # Draw that square using the color it should have in the map
            map.drawSquare(state[0])
            # Remove that subgoal from the plan
            state = state[1:]
            # Redraw the rest of the plan
            map.drawPath(state)
        # Return the current plan and a subgoal in world coordinates
        return (state, map.indicesToPoint(state[0]))

def timeToReplan(plan, currentIndices, map, goalIndices):
    """
    Replan if the current plan is C{None}, if the plan is invalid in
    the map (because it is blocked), or if the plan is empty and we
    are not at the goal (which implies that the last time we tried to
    plan, we failed).
    """
    return plan == None or planInvalidInMap(map, plan, currentIndices) or \
            (plan == [] and not goalIndices == currentIndices) 

def planInvalidInMap(map, plan, currentIndices):
    """
    Checks to be sure all the cells between the robot's current location
    and the first subgoal in the plan are occupiable.
    In low-noise conditions, it's useful to check the whole plan, so failures
    are discovered earlier;  but in high noise, we often have to get
    close to a location before we decide that it is really not safe to
    traverse.

    We actually ignore the case when the robot's current indices are
    occupied;  during mapMaking, we can sometimes decide the robot's
    current square is not occupiable, but we should just keep trying
    to get out of there.
    """
    if len(plan) == 0:
        return False
    wayPoint = plan[0]
    for p in util.lineIndicesConservative(currentIndices, wayPoint)[1:]:
        if not map.robotCanOccupy(p):
            print 'plan invalid', currentIndices, p, wayPoint, '-- replanning'
            return True
    return False
