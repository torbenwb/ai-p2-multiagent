# multiAgents.py
# Code added by Torben Bernhard
# Date / Time: 11/3/2023 12:37
# All tests passing in full on local machine
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        # state s' after taking action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # new position in state s'
        newPos = successorGameState.getPacmanPosition()
        # food before taking action
        currentFood = currentGameState.getFood().asList()
        # food after taking action
        newFood = successorGameState.getFood().asList()
        # new ghost states after taking action
        newGhostStates = successorGameState.getGhostStates()
        # new ghost positions after taking action
        newGhostPositions = successorGameState.getGhostPositions()
        # new scared timers after taking action
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # whether or not ghosts are scared
        scared = newScaredTimes[0] >= 1
        # capsules in the current state before taking action
        capsules = currentGameState.getCapsules()
        score = 0

        # if there is still food
        if len(currentFood) > 0:
            foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in currentFood]
            minFoodDistance = min(foodDistances)
            # prioritize eating an adjacent food pellet over
            # reducing distance to nearby food
            # max food value is 15 - eat and 1 away from another food
            # high score for eating an adjacent food
            if minFoodDistance == 0:
                score += 10
            else:
                # prioritize reducing distance to nearest food
                score += (1 / minFoodDistance)

        # if there are ghosts
        if len(newGhostPositions) > 0:
            # get manhattan distance to each ghost - uses ghost
            # positions after taking action
            ghostDistances = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPositions]
            # minimum distance to ghost
            minGhostDistance = min(ghostDistances)

            if minGhostDistance == 0:
                if scared:
                    # Guaranteed to eat a ghost instead of food
                    score += 15
                else:
                    # Guaranteed to never go to a losing position
                    score -= 100
            else:
                if scared:
                    score += 1 / minGhostDistance
                else:
                    # prioritize distance from nearest ghost
                    # the greater the distance from the nearest ghost
                    # the better - but do not prioritize over food
                    if minGhostDistance > 2:
                        score += 2
                    else:
                        score += minGhostDistance

        if len(capsules) > 0:
            capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in capsules]
            minCapsuleDistance = min(capsuleDistances)

            if minCapsuleDistance == 0:
                score += 20
            else:
                # prioritize reducing distance to capsule over reducing distance to food
                score += (1 / minCapsuleDistance) * 5

        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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

        # given an arbitrary game state - determine the best action

        def node(agent_index, current_state, depth):
            if agent_index >= current_state.getNumAgents():
                agent_index = 0
                depth += 1
            if current_state.isWin() or current_state.isLose():
                return self.evaluationFunction(current_state), None
            if depth >= self.depth:
                return self.evaluationFunction(current_state), None
            if agent_index == 0:
                return max_node(agent_index, current_state, depth)
            else:
                return min_node(agent_index, current_state, depth)

        def max_node(agent_index, current_state, depth):
            v = -999
            a = None
            for action in current_state.getLegalActions(agent_index):
                next_state = current_state.generateSuccessor(agent_index, action)
                u = node(agent_index + 1, next_state, depth)[0]
                if u > v:
                    v = u
                    a = action
            return v, a

        def min_node(agent_index, current_state, depth):
            v = 999
            a = None
            for action in current_state.getLegalActions(agent_index):
                next_state = current_state.generateSuccessor(agent_index, action)
                u = node(agent_index + 1, next_state, depth)[0]
                if u < v:
                    v = u
                    a = action
            return v, a

        return node(0, gameState, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def node(agent_index, current_state, depth, alpha, beta):
            if agent_index >= current_state.getNumAgents():
                agent_index = 0
                depth += 1
            if current_state.isWin() or current_state.isLose():
                return self.evaluationFunction(current_state), None
            if depth >= self.depth:
                return self.evaluationFunction(current_state), None
            if agent_index == 0:
                return max_node(agent_index, current_state, depth, alpha, beta)
            else:
                return min_node(agent_index, current_state, depth, alpha, beta)

        def max_node(agent_index, current_state, depth, alpha, beta):
            v = -999
            a = None
            for action in current_state.getLegalActions(agent_index):
                next_state = current_state.generateSuccessor(agent_index, action)
                u = node(agent_index + 1, next_state, depth, alpha, beta)[0]
                if u > v:
                    if u > beta:
                        return u, a
                    v = u
                    a = action
                    if v > alpha:
                        alpha = v
            return v, a

        def min_node(agent_index, current_state, depth, alpha, beta):
            v = 999
            a = None
            for action in current_state.getLegalActions(agent_index):
                next_state = current_state.generateSuccessor(agent_index, action)
                u = node(agent_index + 1, next_state, depth, alpha, beta)[0]
                if u < v:
                    if u < alpha:
                        return u, a
                    v = u
                    a = action
                    if v < beta:
                        beta = v
            return v, a

        return node(0, gameState, 0, -999, 999)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def node(agent_index, current_state, depth):
            if agent_index >= current_state.getNumAgents():
                agent_index = 0
                depth += 1
            if current_state.isWin() or current_state.isLose():
                return self.evaluationFunction(current_state), None
            if depth >= self.depth:
                return self.evaluationFunction(current_state), None
            if agent_index == 0:
                return max_node(agent_index, current_state, depth)
            else:
                return random_node(agent_index, current_state, depth)

        def max_node(agent_index, current_state, depth):
            v = -999
            a = None
            for action in current_state.getLegalActions(agent_index):
                next_state = current_state.generateSuccessor(agent_index, action)
                u = node(agent_index + 1, next_state, depth)[0]
                if u > v:
                    v = u
                    a = action
            return v, a

        def random_node(agent_index, current_state, depth):
            v = 999
            a = None
            s = 0
            options = []
            for action in current_state.getLegalActions(agent_index):
                next_state = current_state.generateSuccessor(agent_index, action)
                u = node(agent_index + 1, next_state, depth)[0]
                options.append((u, action))
                s += u
            return s / len(options), None

        return node(0, gameState, 0)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Find the following values:
    - evaluated state score - higher is better
    - number of remaining food pellets - lower is better
    - number of remaining capsules - lower is better
    - distance to closest food - lower is better
    - scared timer - higher is better

    Given these values derive a score as a sum of each value
    multiplied by a weight
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    foods = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    numFoods = len(foods)
    numCapsules = len(capsules)
    numGhosts = len(ghosts)

    closeFood = 1
    closeGhost = 2
    scaredTimer = 0

    # distance to nearest food
    if numFoods > 0:
        closeFood = min([manhattanDistance(pos, food) for food in foods])
    # distance to nearest capsule
    if numCapsules > 0:
        closeCapsule = min([manhattanDistance(pos, capsule) for capsule in capsules])
    # distance to nearest ghost
    if numGhosts > 0:
        closeGhost = min([manhattanDistance(pos, ghost) for ghost in ghosts])
    if len(ghostStates) > 0:
        scaredTimer = max([state.scaredTimer for state in ghostStates])

    if closeGhost <= 1:
        closeFood = 99999

    values = [score, numFoods, numCapsules, (1 / closeFood), scaredTimer]
    weights = [1.1, -0.3, -0.25, 1.1, 2]

    sum = 0
    for i in range(0, len(values)):
        sum += values[i] * weights[i]

    return sum


# Abbreviation
better = betterEvaluationFunction
