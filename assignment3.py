"""1. Write a code in Python implementing Value iteration for a grid world given in the
image above
Values you will require:
1. The reward for reaching the goal state = 1
2. The penalty for reaching the red state = -1
3. Step cost = -0.04
4. Probability of going in the direction of the action = 0.7
5. Probability of going in a direction perpendicular to the action = 0.15
Print the utility value of each cell in the grid after each iteration until the values
converge. (Assume the values converge when the difference between the utilities
for each cell is <= 0.0001)
Note: The agent does not change its state if it hits a wall or the boundaries"""

'''
|       | reward + 1 | Penalty -1 |
|-------|------------|------------|
|       |            |            |
|       | wall       |            |
| start |            |            |
'''
