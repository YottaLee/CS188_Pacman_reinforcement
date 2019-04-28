# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            sValue = util.Counter()
            for state in self.mdp.getStates():
                aValue = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    aValue[action] = self.computeQValueFromValues(state, action)
                sValue[state] = aValue[aValue.argMax()]
            for state in self.mdp.getStates():
                self.values[state] = sValue[state]




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        for (nextState, probability) in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += probability*(self.mdp.getReward(state, action, nextState) + self.discount* self.values[nextState])
        return qValue
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None

        aValue = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            aValue[action] = self.computeQValueFromValues(state, action)

        return aValue.argMax()
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        stateList = self.mdp.getStates()
        for i in range(self.iterations):
            state = stateList[i% len(stateList)]
            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                self.values[state] = self.computeQValueFromValues(state,action)




class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # compute predecessors of all states
        statesList = self.mdp.getStates()
        statePredecessors = {}
        queue = util.PriorityQueue()

        for state in statesList:
            statePredecessors[state] = set()

        for state in statesList:
            for action in self.mdp.getPossibleActions(state):
                nStates = self.mdp.getTransitionStatesAndProbs(state, action)
                for nState in nStates:
                    if nState[1] > 0:
                        statePredecessors[nState[0]].add(state)

        for state in statesList:
            if not self.mdp.isTerminal(state):
                qValue = []
                for action in self.mdp.getPossibleActions(state):
                    qValue.append(self.getQValue(state, action))
                diff = abs(max(qValue) - self.values[state])
                queue.push(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            else:
                state = queue.pop()
                qValueState = []
                for action in self.mdp.getPossibleActions(state):
                    qValueState.append(self.getQValue(state,action))
                self.values[state] = max(qValueState)
                for predecessor in statePredecessors[state]:
                    qValuePredecessor = []
                    for action in self.mdp.getPossibleActions(predecessor):
                        qValuePredecessor.append(self.getQValue(predecessor, action))
                    diff = abs(max(qValuePredecessor) - self.values[predecessor])
                    if self.theta < diff:
                        queue.update(predecessor, -diff)

