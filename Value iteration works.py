
'''
World

|       |    + 1     |     - 1    |
|       |            |            |
|       |            |            |
|       |    wall    |            |
| start |            |            |
'''

"""
1. Write a code in Python implementing Value iteration for a grid world given in the
image above
Values you will require:
1. The reward for reaching the goal state = 1
2. The penalty for reaching the red state = -1
3. Step cost = -0.04
4. Probability of going in the direction of the action = 0.7
5. Probability of going in a direction perpendicular to the action = 0.15
6. Discount Factor = 0.95
Print the utility value of each cell in the grid after each iteration until the values
converge. (Assume the values converge when the difference between the utilities
for each cell is <= 0.0001)
Note: The agent does not change its state if it hits a wall or the boundaries
"""

# DEFINING THE MDP

GOAL_STATE = (0, 1)
RED_STATE = (0, 2)
STEP_COST = -0.04
GAMMA = 0.95
DELTA = 0.0001
WALL_STATE = (2, 1)

PROB_ACTION = 0.7
PROB_OTHER_ACTION = 0.15

U = [[0, 1, -1],
     [0 for i in range(3)],
     [0, "WALL", 0],
     [0 for i in range(3)]]

NUM_ROWS = len(U)
NUM_COLS = len(U[0])

ACTIONS = ["DOWN", "LEFT", "UP", "RIGHT"] # Down, Left, Up, Right

# HELPER FUNCTIONS

def get_state_corresponding_to_the_action(r, c, action):
    if action == "UP":
        if r == 0 or (r - 1, c) == WALL_STATE:
            return [r, c]
        return [r - 1, c]
    elif action == "DOWN":
        if r == len(U) - 1 or (r + 1, c) == WALL_STATE:
            return [r, c]
        return [r + 1, c]
    elif action == "LEFT":
        if c == 0 or (r, c - 1) == WALL_STATE:
            return [r, c]
        return [r, c - 1]
    elif action == "RIGHT":
        if c == len(U[0]) - 1 or (r, c + 1) == WALL_STATE:
            return [r, c]
        return [r, c + 1]
    
def get_perpendicular_actions(action):
    if action == "UP" or action == "DOWN":
        return ["LEFT", "RIGHT"]
    elif action == "LEFT" or action == "RIGHT":
        return ["UP", "DOWN"]
    
def get_possible_other_states(r, c, action):
    possible_other_states = []
    for action in get_perpendicular_actions(action):
        possible_other_states.append(get_state_corresponding_to_the_action(r, c, action))
    return possible_other_states

def get_value_state(U, r, c, action):
    value_state = 0

    x, y = get_state_corresponding_to_the_action(r, c, action)
    value_state += PROB_ACTION * (GAMMA * U[x][y])

    for other_state in get_possible_other_states(r, c, action):
        value_state += PROB_OTHER_ACTION * ( GAMMA * U[other_state[0]][other_state[1]])
    value_state += STEP_COST
    return value_state

# MAIN FUNCTIONS

def value_iteration(U):
    i = 0
    while True:
        nextU =[[0, 1, -1],
                [0 for i in range(3)],
                [0, "WALL", 0],
                [0 for i in range(3)]]
        delta = 0
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                if (r, c) == RED_STATE or (r, c) == WALL_STATE or (r, c) == GOAL_STATE:
                    continue
                utility = []
                for action in ACTIONS:
                    utility.append(get_value_state(U, r, c, action))
                nextU[r][c] = max(utility)
                delta = max(delta, abs(U[r][c] - nextU[r][c]))
        U = nextU
        print("Iteration: ", i+1, end='\n\n')
        i += 1
        STR_U = [[str(U[r][c]) for c in range(NUM_COLS)] for r in range(NUM_ROWS)]
        STR_U[GOAL_STATE[0]][GOAL_STATE[1]] = "+1"
        print_list_of_list(STR_U)
        if delta < DELTA:
            break
    return U

# find the optimal policy
def get_policy(U):
    policy = [[0 for i in range(NUM_COLS)] for j in range(NUM_ROWS)]
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            if (r, c) == RED_STATE or (r, c) == WALL_STATE or (r, c) == GOAL_STATE:
                continue
            utility = {}
            for action in ACTIONS:
                utility[action] = get_value_state(U, r, c, action) # dicitionary
            policy[r][c] = max(utility, key=utility.get)
    return policy

# PRINTING FUNCTION
def print_list_of_list(list_of_list):
    n = max(len(x) for l in list_of_list for x in l)

    for row in list_of_list:
        print('|', end=' ')
        for x in row:
            print(" "+''.join(x.ljust(n + 2) + '|'), end='')
        print("\n")
    
if __name__ == "__main__":

    U = value_iteration(U)

    print("Utility:", end='\n\n')
    STR_U = [[str(U[r][c]) for c in range(NUM_COLS)] for r in range(NUM_ROWS)]
    STR_U[GOAL_STATE[0]][GOAL_STATE[1]] = "+1"
    print_list_of_list(STR_U)

    # Print the policy
    policy = get_policy(U)
    print("Policy:", end='\n\n')

    policy[GOAL_STATE[0]][GOAL_STATE[1]] = "+1"
    policy[RED_STATE[0]][RED_STATE[1]] = "-1"
    policy[WALL_STATE[0]][WALL_STATE[1]] = "WALL"

    print_list_of_list(policy)
