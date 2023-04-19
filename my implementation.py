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
    
def isValidAction(state, action):
    if action == 'UP':
        return state[0] > 0
    elif action == 'DOWN':
        return state[0] < 3
    elif action == 'LEFT':
        return state[1] > 0
    elif action == 'RIGHT':
        return state[1] < 2
    
# write the reward function here
def reward(state):
    if state == GOAL_STATE:
        return REWARD_GOAL
    elif state == RED_STATE:
        return REWARD_RED
    else:
        return STEP_COST

def calculate_utility(state, action, U):
    utility = 0
    utility += PROB_ACTION * (reward(transition(state, action)) + DISCOUNT_FACTOR * U[transition(state, action)])
    perp_action = []
    if action == 'UP' or action == 'DOWN':
        perp_action.append('LEFT')
        perp_action.append('RIGHT')
    if action == 'LEFT' or action == 'RIGHT':
        perp_action.append('UP')
        perp_action.append('DOWN')
    for paction in perp_action:
        utility += PROB_PERPENDICULAR_ACTION * (reward(transition(state, paction)) + DISCOUNT_FACTOR * U[transition(state, paction)])
    return utility



def value_iteration():
    # write your code here
    U = np.zeros((4, 3))
    k = 0
    while True:
        U_new = np.zeros((4, 3))
        for i in range(4):
            for j in range(3):
                if (i, j) == WALL:
                    continue
                max_utility = -float('inf')
                for action in ACTIONS:
                    if isValidAction((i, j), action):
                        utility = calculate_utility((i, j), action, U)
                        if utility > max_utility:
                            max_utility = utility
                U_new[i, j] = max_utility
        # print("Iteration: ", k + 1)
        # k += 1
        # print(U_new)
        if np.max(np.abs(U_new - U)) < CONVERGENCE_THRESHOLD:
            break
        U = U_new
        print("Iteration: ", k + 1)
        k += 1
        print(U)
    return U

def optimal_policy(U):
    # write your code here
    # policy = np.zeros((4, 3), dtype = object) # to print the whole string ('UP') instead of just the first letter ('U')
    policy = np.zeros((4, 3), dtype = str)
    for i in range(4):
        for j in range(3):
            if (i, j) == WALL:
                continue
            max_utility = -float('inf')
            for action in ACTIONS:
                if isValidAction((i, j), action):
                    utility = calculate_utility((i, j), action, U)
                    if utility > max_utility:
                        max_utility = utility
                        policy[i, j] = action
                        print(action)
    return policy

if __name__ == '__main__':
    U = value_iteration()
    print(U)
    policy = optimal_policy(U)
    print(policy)