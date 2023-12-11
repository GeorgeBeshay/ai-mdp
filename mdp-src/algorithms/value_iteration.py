from typing import List, Tuple
# import numpy as np


def get_successors(currX, currY, action, row_length=3, col_length=3) -> List[Tuple[int, int]]:
    # N - W - S - E
    x = []
    if action == 'N':
        x = [(-1, 0), (0, -1), (0, 1)]
    elif action == 'W':
        x = [(0, -1), (1, 0), (-1, 0)]
    elif action == 'S':
        x = [(1, 0), (0, -1), (0, 1)]
    elif action == 'E':
        x = [(0, 1), (1, 0), (-1, 0)]
    else:
        return []  # error should not occur

    sol = []
    for (i, j) in x:
        if currX + i < row_length and currY + j < col_length:
            sol.append((currX + i, currY + j))
        else:
            sol.append((currX, currY))
    return sol


# print(get_successors(1, 1, 'N'))


def is_terminal_state(board, i, j):
    return board[i][j] != -1


def are_equal(V_k, V_k_next, epsilon=1e-6) -> bool:
    # Check if the dimensions are the same
    if len(V_k) != len(V_k_next) or any(len(row) != len(V_k_next[0]) for row in V_k):
        return False

    # Check element-wise equality within epsilon
    for i in range(len(V_k)):
        for j in range(len(V_k[0])):
            if abs(V_k[i][j] - V_k_next[i][j]) > epsilon:
                return False

    return True


def value_iteration(board: List[List[int]], V_k: List[List[float]], gamma=0.99, k=0) -> Tuple[List[List[float]], List[List[str]]]:
    print(f"================================================= k = {k} =================================================")
    num_rows = len(board)
    num_columns = len(board[0])
    V_k_next = [[0] * num_columns for _ in range(num_rows)]
    arg_max = [[''] * num_columns for _ in range(num_rows)]

    for i in range(num_rows):
        for j in range(num_columns):
            if is_terminal_state(board, i, j):  # You need to define is_terminal_state
                V_k_next[i][j] = board[i][j]  # Terminal state value
                arg_max[i][j] = 'R'  # No action for terminal state
                continue
            max_next_V = -float('inf')
            max_arg_s = None
            for action in ['N', 'W', 'S', 'E']:
                sum = 0.0
                first_successor = True
                # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                # print(f"i = {i}, j = {j}")
                # print("action is = ", action)
                for successor in get_successors(i, j, action):
                    (actX, actY) = successor
                    # print("successor is = ", successor)
                    transition = 0.8 if first_successor else 0.1
                    sum += transition * (board[i][j] + gamma * V_k[actX][actY])
                    # print(f"sum += {transition} * (board[{actX}][{actY}] + gamma * V_k[{actX}][{actY}])"
                    #       f" = sum += {transition} * ({board[actX][actY]} + {gamma} * {V_k[actX][actY]})")
                    # print("sum = ", sum)
                    first_successor = False
                if sum > max_next_V:
                    max_next_V = sum
                    max_arg_s = action
            # print(f"max sum for state = board[{i}][{j}] is {max_next_V} and action is '{max_arg_s}'")
            V_k_next[i][j] = max_next_V
            arg_max[i][j] = max_arg_s

    for row in V_k_next:
        print(" ".join("{:.2f}".format(value) for value in row))
    for row in arg_max:
        print(" ".join(row))

    if are_equal(V_k, V_k_next):
        return V_k_next, arg_max
    else:
        return value_iteration(board, V_k_next,k=k+1)


# Initialize board and V_k
example_board = [[0, -1, 10], [-1, -1, -1], [-1, -1, -1]]
V_0 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

# Call the value_iteration function
V_k, pi_k = value_iteration(example_board, V_0)
# Print V_k
print("V_k =")
for row in V_k:
    print(" ".join("{:.2f}".format(value) for value in row))

# Print pi_k
print("pi_k =")
for row in pi_k:
    print(" ".join(row))
