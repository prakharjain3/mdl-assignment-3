'''
|       | reward + 1 | Penalty -1 |
|-------|------------|------------|
|       |  &nbsp;    |  &nbsp;    |
|       | wall       |            |
| start |            |            |
'''

'''
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
Print the optimal policy for each cell in the grid.
Note: The agent does not change its state if it hits a wall or the boundaries
'''

import numpy as np

GOAL_STATE = (0, 1)
RED_STATE = (0, 2)
WALL = (2, 1)

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
PROB_ACTION = 0.7
PROB_PERPENDICULAR_ACTION = 0.15

REWARD_GOAL = 1
REWARD_RED = -1
STEP_COST = -0.04

DISCOUNT_FACTOR = 0.95
CONVERGENCE_THRESHOLD = 0.0001

def transition(state, action):
    if action == 'UP':
        if state[0] == 0:
            return state
        return (state[0] - 1, state[1])
    elif action == 'DOWN':
        if state[0] == 3:
            return state
        return (state[0] + 1, state[1])
    elif action == 'LEFT':
        if state[1] == 0:
            return state
        return (state[0], state[1] - 1)
    elif action == 'RIGHT':
        if state[1] == 2:
            return state
        return (state[0], state[1] + 1)

def reward(state):
    if state == GOAL_STATE:
        return REWARD_GOAL
    elif state == RED_STATE:
        return REWARD_RED
    else:
        return STEP_COST
    
def value_iteration():
    # initialize the utility function
    U = np.zeros((4, 3))
    U[GOAL_STATE] = REWARD_GOAL
    U[RED_STATE] = REWARD_RED
    U[WALL] = None

    # initialize the policy function
    pi = np.zeros((4, 3), dtype='object')
    pi[GOAL_STATE] = None
    pi[RED_STATE] = None
    pi[WALL] = None

    # initialize the delta
    delta = 1

    # start the iteration
    while delta > CONVERGENCE_THRESHOLD:
        delta = 0
        for i in range(4):
            for j in range(3):
                if (i, j) == GOAL_STATE or (i, j) == RED_STATE or (i, j) == WALL:
                    continue
                else:
                    # calculate the utility
                    U_old = U[i, j]
                    U_max = -np.inf
                    for action in ACTIONS:
                        U_max = max(U_max, sum([PROB_ACTION * U[transition((i, j), action)] if transition((i, j), action) == transition((i, j), a) else PROB_PERPENDICULAR_ACTION * U[transition((i, j), action)] for a in ACTIONS]))
                    U[i, j] = reward((i, j)) + DISCOUNT_FACTOR * U_max
                    delta = max(delta, abs(U_old - U[i, j]))
                    # calculate the policy
                    pi[i, j] = ACTIONS[np.argmax([sum([PROB_ACTION * U[transition((i, j), action)] if transition((i, j), action) == transition((i, j), a) else PROB_PERPENDICULAR_ACTION * U[transition((i, j), action)] for a in ACTIONS]) for action in ACTIONS])]
        print(U)
    print(pi)

value_iteration()