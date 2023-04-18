import numpy as np

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

START_STATE = (3, 0)
CURRENT_STATE = START_STATE
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

# write the transition function here
def transition(state, action):
    if action == 'UP':
        return (state[0] - 1, state[1])
    elif action == 'DOWN':
        return (state[0] + 1, state[1])
    elif action == 'LEFT':
        return (state[0], state[1] - 1)
    elif action == 'RIGHT':
        return (state[0], state[1] + 1)
    
# write the reward function here
def reward(state):
    if state == GOAL_STATE:
        return REWARD_GOAL
    elif state == RED_STATE:
        return REWARD_RED
    else:
        return STEP_COST

# write the value iteration function here
def value_iteration():
    # initialize the utility values to 0
    utility_values = np.zeros((4, 3))
    utility_values[GOAL_STATE] = REWARD_GOAL
    utility_values[RED_STATE] = REWARD_RED
    utility_values[WALL] = np.nan
    
    # initialize the policy to None
    policy = np.full((4, 3), None)
    policy[GOAL_STATE] = 'GOAL'
    policy[RED_STATE] = 'RED'
    policy[WALL] = 'WALL'
    
    # initialize the delta to a large value
    delta = 100
    
    # loop until delta is less than the convergence threshold
    while delta > CONVERGENCE_THRESHOLD:
        # initialize the delta to 0
        delta = 0
        
        # loop through all the states
        for i in range(4):
            for j in range(3):
                # skip the wall state
                if (i, j) == WALL:
                    continue
                
                # get the current utility value
                current_utility_value = utility_values[i, j]
                
                # initialize the utility value to 0
                utility_value = 0
                
                # loop through all the actions
                for action in ACTIONS:
                    # get the next state
                    next_state = transition((i, j), action)
                    
                    # check if the next state is a wall
                    if next_state == WALL:
                        # if the next state is a wall, then the utility value is the current utility value
                        utility_value += utility_values[i, j] * PROB_ACTION
                    else:
                        # if the next state is not a wall, then the utility value is the reward + the discounted utility value of the next state
                        utility_value += reward(next_state) + DISCOUNT_FACTOR * utility_values[next_state[0], next_state[1]] * PROB_ACTION
                    
                    # get the perpendicular actions
                    perpendicular_actions = [a for a in ACTIONS if a != action]
                    
                    # loop through all the perpendicular actions
                    for perpendicular_action in perpendicular_actions:
                        # get the next state
                        next_state = transition((i, j), perpendicular_action)
                        
                        # check if the next state is a wall
                        if next_state == WALL:
                            # if the next state is a wall, then the utility value is the current utility value
                            utility_value += utility_values[i, j] * PROB_PERPENDICULAR_ACTION
                        else:
                            # if the next state is not a wall, then the utility value is the reward + the discounted utility value of the next state
                            utility_value += reward(next_state) + DISCOUNT_FACTOR * utility_values[next_state[0], next_state[1]] * PROB_PERPENDICULAR_ACTION
                            