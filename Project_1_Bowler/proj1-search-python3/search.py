# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    frontier = util.Stack()
    start = problem.getStartState()

    "Keep track of the (x,y) location and the directions required to get there"
    frontier.push((start, []))

    "In this search, explored contains all nodes popped from stack"
    explored = []

    while not frontier.isEmpty():

        popped = frontier.pop()

        "Re-visit protection"
        if not popped[0] in explored:
            explored.append(popped[0])

            # print("EXPLORED", explored)
            if problem.isGoalState(popped[0]):
                "Return solution of actions"
                return popped[1]

            successors = problem.getSuccessors(popped[0])
            for s in successors:
                "Add direction of arrival at new node"
                # print("\n\nPopped", popped[1], "S", s[1])

                "Deep copy, as this blasted language assigns references by default, "
                "and I DO NOT want to edit popped[1]"
                dirs = popped[1].copy()

                "Actions to new node"
                dirs.append(s[1])

                # print("DIRS",dirs)
                frontier.push((s[0], dirs))

    "Frontier is empty, return failure"
    return []



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue()

    start = problem.getStartState()
    "Again, keeping track of location and directions"
    frontier.push((start, []))

    "In this search, enqueued contains all in the queue as well as popped"
    enqueued = []

    enqueued.append(start)

    while not frontier.isEmpty():
        popped = frontier.pop()
        # print("\n\nPOPPED:", popped)

        if problem.isGoalState(popped[0]):
            "Return solution"
            return popped[1]

        successors = problem.getSuccessors(popped[0])
        for s in successors:
            "Check for re-visit BEFORE enqueue since we are keeping track of enqueued nodes,"
            "versus after pop"
            if not s[0] in enqueued:
                "Use enqueued for children since the util.Queue doesn't have a contains method"
                "If it did, we could keep track of visited nodes separately, as in the pseudocode from the book,"
                "but this works"

                enqueued.append(s[0])
                "Add direction of arrival at new node"

                "Deep copy, as this blasted language assigns references by default, "
                "and I DO NOT want to edit popped[1]"
                dirs = popped[1].copy()
                dirs.append(s[1])
                # print("DIRS",dirs)
                frontier.push((s[0], dirs))

    "Frontier is empty, return failure"
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()

    start = problem.getStartState()

    "Here, I keep track of a cost on the frontier, but I also keep track of the priority"
    "for the queue, which includes the cost of the heuristic at that node. In this case, they are"
    "the same. but keeping track of the cost on the frontier in this way allows for easier access"
    "of the cost it took to arrive at the new node."
    "We start with the nullHeuristic = 0"
    frontier.push((start, [], nullHeuristic(start)), nullHeuristic(start))

    "Explored in this problem is all nodes popped off the queue"
    explored = []

    while not frontier.isEmpty():
        popped = frontier.pop()

        "Check after pop as in dfs"
        if not popped[0] in explored:
            explored.append(popped[0])

            if problem.isGoalState(popped[0]):
                "Return solution"
                return popped[1]

            successors = problem.getSuccessors(popped[0])
            for s in successors:

                "Get priority for queue: The cost of this node + cost it took to get here"
                priority = s[2] + popped[2]

                "Add direction of arrival at new node"

                "Again, deep copy...."
                dirs = popped[1].copy()
                dirs.append(s[1])

                "Use UPDATE so if the value is already in the queue, it's priority is updated"
                frontier.update((s[0], dirs, priority), priority)

    "Frontier is empty, return failure"
    return []



def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()

    start = problem.getStartState()

    "As in UCS, I keep track of a cost on the frontier, but I also keep track of the priority"
    "for the queue, which includes the cost of the heuristic at that node."
    "Keeping track of the cost on the frontier in this way allows for easier access of the cost"
    "it took to arrive at the new node."

    frontier.push((start, [], heuristic(start, problem)), heuristic(start, problem))

    "Explored in this problem is all nodes popped off the queue"
    explored = []

    while not frontier.isEmpty():
        popped = frontier.pop()

        "Check after pop as in dfs and UCS"
        if not popped[0] in explored:
            explored.append(popped[0])

            if problem.isGoalState(popped[0]):
                "Return solution"
                # print("------Solution Found --------")
                return popped[1]

            successors = problem.getSuccessors(popped[0])
            for s in successors:
                "Get heuristic of node"
                # print("SUCCESSOR", s[0])

                "Get priority for queue: The cost of this node + cost it took to get here + heuristic"
                cost = s[2] + popped[2]
                h = heuristic(s[0], problem)
                priority = h + cost
                # print("HEUR:",h,"PRIORITY",priority)

                "Add direction of arrival at new node"

                "Again, deep copy..."
                dirs = popped[1].copy()
                dirs.append(s[1])

                "Use UPDATE so if the value is already in the queue, it's priority is updated"
                frontier.update((s[0], dirs, cost), priority)

    "Frontier is empty, return failure"
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
