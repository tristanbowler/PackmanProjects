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
        """*** YOUR CODE HERE ***"""
        for i in range(0, self.iterations):
            # Create a new counter
            state_counter = util.Counter()
            states = self.mdp.getStates()

            for state in states:
                if not self.mdp.isTerminal(state):  # This just saves a little bit of memory
                    # Create another new counter
                    actions_counter = util.Counter()
                    actions = self.mdp.getPossibleActions(state)

                    for action in actions:
                        # Set Q Value for action
                        actions_counter[action] = self.computeQValueFromValues(state, action)

                    # argMax returns the key with the biggest value in the counter
                    max_action = actions_counter.argMax()
                    state_counter[state] = actions_counter[max_action]  # Max Q Value

            # Update the values list before the next iteration
            self.values = state_counter


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
        q = 0.0

        # Transition states and probs return type:
        "Returns list of(nextState, prob) pairs representing the states reachable from " \
            " 'state' by taking 'action' along with their transition probabilities."
        transition_states = self.mdp.getTransitionStatesAndProbs(state, action)  # list(tuple(nextState, prob))

        for next_state, probability in transition_states:

            # Apply discount
            discounted_value = self.discount * self.values[next_state]

            # Get reward
            "Get reward for state, action, nextState transition. Note that the reward depends only on the state " \
                "being departed (as in the R+N book examples, which more or less use this convention)."
            reward = self.mdp.getReward(state, action, next_state)  # float

            q += probability * (discounted_value + reward)

        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        actions_counter = util.Counter()
        actions = self.mdp.getPossibleActions(state)

        for action in actions:
            actions_counter[action] = self.computeQValueFromValues(state, action)

        return actions_counter.argMax()

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

        states = self.mdp.getStates()

        for i in range(0, self.iterations):
            # this is the key. It makes the algorithm rotate through the states for the number of iterations
            # instead of doing all of the states * the number of iterations
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                # Same as above
                if not self.mdp.isTerminal(state):  # This just saves a little bit of memory
                    # Create another new counter
                    actions_counter = util.Counter()
                    actions = self.mdp.getPossibleActions(state)

                    for action in actions:
                        # Set Q Value for action
                        actions_counter[action] = self.computeQValueFromValues(state, action)

                    # argMax returns the key with the biggest value in the counter
                    max_action = actions_counter.argMax()
                    self.values[state] = actions_counter[max_action]



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
        queue = util.PriorityQueue()
        state_counter = util.Counter()
        states = self.mdp.getStates()

        predecessors = {}  # dict of <state, list([state])>

        for state in states:
            if not self.mdp.isTerminal(state):
                actions_counter = util.Counter()
                actions = self.mdp.getPossibleActions(state)

                # This sets up the dict of predecessors for each action state
                for action in actions:
                    transition_states = self.mdp.getTransitionStatesAndProbs(state, action)
                    for transition_state, probability in transition_states:
                        if transition_state in predecessors:
                            predecessors[transition_state].append(state)
                        else:
                            predecessors[transition_state] = [state]
                    # Sets the Q Value
                    actions_counter[action] = self.computeQValueFromValues(state, action)

                # argMax returns the key with the biggest value in the counter
                max_action = actions_counter.argMax()
                diff = abs(self.values[state] - actions_counter[max_action]) # Max Q Value
                # -diff because min queue and we want max
                queue.update(state, -diff)

        for i in range(0, self.iterations):
            if not queue.isEmpty():
                current = queue.pop()
                if not self.mdp.isTerminal(current):
                    actions_counter = util.Counter()
                    actions = self.mdp.getPossibleActions(current)

                    # Get max Q Value of action states
                    for action in actions:
                        # Set Q Value for action
                        actions_counter[action] = self.computeQValueFromValues(current, action)

                    # argMax returns the key with the biggest value in the counter
                    max_action = actions_counter.argMax()
                    # Update value
                    self.values[current] = actions_counter[max_action]

                    for predecessor in predecessors:
                        if not self.mdp.isTerminal(predecessor):
                            predecessor_actions_counter = util.Counter()
                            predecessor_actions = self.mdp.getPossibleActions(predecessor)

                            # Find max predecessor Q value
                            for predecessor_action in predecessor_actions:
                                # Set Q Value for action
                                predecessor_actions_counter[predecessor_action] = self.computeQValueFromValues(predecessor, predecessor_action)
                            predecessor_max_action = predecessor_actions_counter.argMax()

                            diff = abs(self.values[predecessor] - predecessor_actions_counter[predecessor_max_action])
                            # Add back on to queue is the difference is higher than our tolerance
                            if diff > self.theta:
                                queue.update(predecessor, -diff)
