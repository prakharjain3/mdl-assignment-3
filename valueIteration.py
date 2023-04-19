# Arguments
REWARD = -0.04 # constant reward for non-terminal states
DISCOUNT = 0.95
MAX_ERROR = 0.0001

# Set up the initial environment
NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
NUM_ROW = 4
NUM_COL = 3
U = [[0, 1, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

WALL_STATE = (2, 1)
GOAL_STATE = (0, 1)
PENALTY_STATE = (0, 2)

# Visualization
def printEnvironment(arr, policy=False):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            if (r, c) == WALL_STATE:
                val = "WALL"
            elif (r, c) == GOAL_STATE:
                val = "+1"
            elif (r, c) == PENALTY_STATE:
                val = "-1"
            else:
                if policy:
                    val = ["Down", "Left", "Up", "Right"][arr[r][c]]
                else:
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL or (newR, newC) == WALL_STATE: # collide with the boundary or the wall
        return U[r][c]
    else:
        return U[newR][newC]

# Calculate the utility of a state given an action
def calculateU(U, r, c, action):
    u = REWARD
    u += 0.15 * DISCOUNT * getU(U, r, c, (action-1)%4)
    u += 0.7 * DISCOUNT * getU(U, r, c, action)
    u += 0.15 * DISCOUNT * getU(U, r, c, (action+1)%4)
    return u

def valueIteration(U):
    print("During the value iteration:\n")
    while True:
        nextU = [[0, 1, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r, c) == WALL_STATE or (r, c) == GOAL_STATE or (r, c) == PENALTY_STATE:
                    continue
                nextU[r][c] = max([calculateU(U, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        printEnvironment(U)
        if error < MAX_ERROR * (1-DISCOUNT) / DISCOUNT:
            break
    return U

# Get the optimal policy from U
def getOptimalPolicy(U):
    policy = [[-1, -1, -1] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            if (r, c) == WALL_STATE or (r, c) == GOAL_STATE or (r, c) == PENALTY_STATE:
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy

# Print the initial environment
print("The initial U is:\n")
printEnvironment(U)

# Value iteration
U = valueIteration(U)

# Get the optimal policy from U and print it
policy = getOptimalPolicy(U)
print("The optimal policy is:\n")
printEnvironment(policy, True)
