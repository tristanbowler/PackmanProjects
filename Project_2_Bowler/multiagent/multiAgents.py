# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        isScared = False

        "There will be a ghost there"
        if successorGameState.isLose():
            return -999_999

        "We will eat all the dots there"
        if successorGameState.isWin():
            return 999_999

        "Find closest ghost"
        closeGhost = 999_999
        for i in range(0,len(newGhostStates)):
            if newScaredTimes[i] == 0:
                isScared = False
            else:
                isScared = True

            ghostPos = successorGameState.getGhostPosition(i+1);
            ghostDist = manhattanDistance(ghostPos, newPos)
            if ghostDist < closeGhost:
                closeGhost = ghostDist

        "Find closest food"
        closeFood = 999_999
        for foodPos in newFood:
            foodDist = manhattanDistance(foodPos, newPos)
            if foodDist < closeFood:
                closeFood = foodDist

        "Score of the action state"
        score = scoreEvaluationFunction(successorGameState)

        if not isScared:
            "Score + weight towards food - weight away from scary ghost"
            return score + 8.0 / closeFood - 16.0 / closeGhost
        else:
            "Score + weight towards food + weight toward scared ghost"
            return score + 8.0 / closeFood + 16.0 / closeGhost


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    pacman = 0

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        maxScore, maxAction = self.minimax(gameState, self.depth, self.pacman)
        return maxAction

    def minimax(self, gameState, depth, agent):
        "All the ghosts have had their turn, pacman's turn and move down a level in the tree"
        numGhosts = gameState.getNumAgents() - 1
        if agent > numGhosts:
            depth = depth - 1
            agent = self.pacman

        if depth is 0 or (gameState.isLose() or gameState.isWin()):
            "Made it down to a leaf or an end state"
            return tuple((self.evaluationFunction(gameState), ""))

        return self.takeTurn(gameState, depth, agent)

    def takeTurn(self, gameState, depth, agent):

        if agent is self.pacman:
            extremeScore = -999_999
        else:
            extremeScore = 999_999

        extremeAction = None
        actions = gameState.getLegalActions(agent)

        for action in actions:
            "Generate all legal actions from the pacman moves"
            successorGameState = gameState.generateSuccessor(agent, action)
            score, act = self.minimax(successorGameState, depth, agent + 1)
            if agent is self.pacman:
                "Maximizing"
                if score > extremeScore:
                    extremeScore = score
                    extremeAction = action
            else:
                if score < extremeScore:
                    "Minimizing"
                    extremeScore = score
                    extremeAction = action

        return tuple((extremeScore, extremeAction))



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    pacman = 0

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxScore, maxAction = self.minimax(gameState, self.depth, self.pacman, -999_999, 999_999)
        return maxAction


    def minimax(self, gameState, depth, agent, a, b):
        "All the ghosts have had their turn, pacman's turn and move down a level in the tree"
        numGhosts = gameState.getNumAgents() - 1
        if agent > numGhosts:
            depth = depth - 1
            agent = self.pacman

        if depth is 0 or (gameState.isLose() or gameState.isWin()):
            "Made it down to a leaf or an end state"
            return tuple((self.evaluationFunction(gameState), ""))

        return self.takeTurn(gameState, depth, agent, a, b)

    def takeTurn(self, gameState, depth, agent, a, b):
        if agent is self.pacman:
            extremeScore = -999_999
        else:
            extremeScore = 999_999

        extremeAction = None
        actions = gameState.getLegalActions(agent)

        for action in actions:
            "Generate all legal actions from the pacman moves"
            successorGameState = gameState.generateSuccessor(agent, action)
            score, act = self.minimax(successorGameState, depth, agent + 1, a, b)

            if agent is self.pacman:
                "Maximizing"
                if score > extremeScore:
                    extremeScore = score
                    extremeAction = action
                if extremeScore > b:
                    "Beta check"
                    return tuple((extremeScore, extremeAction))
                "Update alpha"
                a = max(a, score)

            else:
                if score < extremeScore:
                    "minimizing"
                    extremeScore = score
                    extremeAction = action
                if extremeScore < a:
                    "alpha check"
                    return tuple((extremeScore, extremeAction))
                "Update beta"
                b = min(b, score)
        return tuple((extremeScore, extremeAction))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    pacman = 0

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxScore, maxAction = self.expectimax(gameState, self.depth, self.pacman)
        return maxAction

    def expectimax(self, gameState, depth, agent):
        "All the ghosts have had their turn, pacman's turn and move down a level in the tree"
        numGhosts = gameState.getNumAgents() - 1
        if agent > numGhosts:
            depth = depth - 1
            agent = self.pacman

        if depth == 0 or (gameState.isLose() or gameState.isWin()):
            "Made it down to a leaf or an end state"
            return tuple((self.evaluationFunction(gameState), ""))

        return self.takeTurn(gameState, depth, agent)

    def takeTurn(self, gameState, depth, agent):

        if agent is self.pacman:
            extremeScore = -999_999
        else:
            extremeScore = 0

        extremeAction = None
        actions = gameState.getLegalActions(agent)

        "Used in expecti portion"
        probability = (1.0 / len(actions))

        for action in actions:
            "Generate all legal actions from the pacman moves"
            successorGameState = gameState.generateSuccessor(agent, action)
            score, act = self.expectimax(successorGameState, depth, agent + 1)
            if agent is self.pacman:
                "Find the max score of all actions"
                if score > extremeScore:
                    extremeScore = score
                    extremeAction = action
            else:
                "Find the average value of the actions"
                extremeScore = extremeScore + probability * score
                extremeAction = None

        return tuple((extremeScore, extremeAction))




def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I very closely followed my reactive agent code from above.

    First I take a look at if the game state will result in a win or a loss.
    If not then I find the closest food and the closest ghost.

    The weights are such that we want to move towards the food (positive)
    and away from the ghost (negative)

    Further, if the ghost is farther away, we rate them with less importance
    same for food. The closer the food the more motivated he is to get it,
    and the closer the ghost the more motivated he is to stay away.

    Finally, if the ghosts are scared, then pacman chases them down, maximizing
    his score just a little bit more.
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    isScared = False

    "There will be a ghost there"
    if currentGameState.isLose():
        return -999_999

    "We will eat all the dots there"
    if currentGameState.isWin():
        return 999_999

    "Find closest ghost"
    closeGhost = 999_999
    for i in range(0, len(newGhostStates)):
        "Is the closest ghost scared?"
        if newScaredTimes[i] == 0:
            isScared = False
        else:
            isScared = True

        ghostPos = currentGameState.getGhostPosition(i + 1);
        ghostDist = manhattanDistance(ghostPos, newPos)
        if ghostDist < closeGhost:
            closeGhost = ghostDist

    "Find closest food"
    closeFood = 999_999
    for foodPos in newFood:
        foodDist = manhattanDistance(foodPos, newPos)
        if foodDist < closeFood:
            closeFood = foodDist

    score = currentGameState.getScore()

    if not isScared:
        "Score + weight towards food - weight away from scary ghost"
        return score + 8.0 / closeFood - 16.0 / closeGhost
    else:
        "Score + weight towards food + weight toward scared ghost"
        return score + 8.0 / closeFood + 16.0 / closeGhost


# Abbreviation
better = betterEvaluationFunction
