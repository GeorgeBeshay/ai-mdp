from typing import List, Tuple
import numpy as np


def get_next_moves(currX: int, currY: int, action: str, row_length: int = 3, col_length: int = 3) -> List[
    Tuple[int, int]]:
    """
        Get the next moves based on the current position and action.

        :param currX: Current X-coordinate.
        :param currY: Current Y-coordinate.
        :param action: Action ('N', 'W', 'S', 'E').
        :param row_length: Number of rows in the world.
        :param col_length: Number of columns in the world.

        :return: List of tuples representing the next moves.
    """
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
        return []  # error: This should not occur

    next_moves = []
    for (i, j) in x:
        if 0 <= currX + i < row_length and 0 <= currY + j < col_length:
            next_moves.append((currX + i, currY + j))
        else:
            next_moves.append((currX, currY))
    return next_moves


def is_terminal_state(rewards: List[List[int]], i: int, j: int) -> bool:
    """
        Check if the given position represents a terminal state in the world.

        :param rewards: 2D list representing the game world.
        :param i: Row index.
        :param j: Column index.

        :return: True if the position is a terminal state, False otherwise.
        """
    return rewards[i][j] != -1


def check_value_convergence(V_k: List[List[float]], V_k_next: List[List[float]], epsilon: float = 1e-6) -> bool:
    """
        Check if the values have converged between two iterations.

        :param V_k: Current values.
        :param V_k_next: Values from the next iteration.
        :param epsilon: Convergence threshold.

        :return: True if the values have converged, False otherwise.
        """
    # Check if the dimensions are the same
    if len(V_k) != len(V_k_next) or any(len(row) != len(V_k_next[0]) for row in V_k):
        return False

    # Check element-wise equality within epsilon
    for i in range(len(V_k)):
        for j in range(len(V_k[0])):
            if abs(V_k[i][j] - V_k_next[i][j]) > epsilon:
                return False
    return True


def value_iteration(world: List[List[int]], V_k: List[List[float]], gamma: float = 0.99) -> Tuple[
    List[List[float]], List[List[str]]]:
    """
        Perform value iteration to calculate the optimal values and policies for each state.

        :param world: 2D list representing the game world.
        :param V_k: Current values.
        :param gamma: Discount factor.

        :return: Tuple containing the updated values and optimal policies.
    """
    num_rows = len(world)
    num_columns = len(world[0])

    k = 0
    max_iterations = 100  # Maximum number of iterations to avoid infinite loop

    while True:
        V_k_next = [[0] * num_columns for _ in range(num_rows)]
        arg_max = [[''] * num_columns for _ in range(num_rows)]

        for i in range(num_rows):
            for j in range(num_columns):
                if is_terminal_state(world, i, j):
                    V_k_next[i][j] = world[i][j]  # Terminal state value
                    arg_max[i][j] = 'R'  # No action for terminal state
                    continue
                max_next_V = -float('inf')
                max_arg_s = None
                for action in ['N', 'W', 'S', 'E']:
                    sum = 0.0
                    first_successor = True
                    for successor in get_next_moves(i, j, action):
                        (actX, actY) = successor
                        transition = 0.8 if first_successor else 0.1
                        sum += transition * (world[i][j] + gamma * V_k[actX][actY])
                        first_successor = False
                    if sum > max_next_V:
                        max_next_V = sum
                        max_arg_s = action

                V_k_next[i][j] = max_next_V
                arg_max[i][j] = max_arg_s

        if check_value_convergence(V_k, V_k_next):
            break  # Exit the loop if values have converged

        # Making the V_k be the V_k_next
        V_k = [i[:] for i in V_k_next]

        k += 1
        if k >= max_iterations:
            break  # Exit the loop if the maximum number of iterations is reached

    return V_k_next, arg_max, k

# ------------------------------------ Separator -------------------------------------------------
# Main Program

# Set print options for NumPy arrays
np.set_printoptions(5, suppress=True)

while True:
    r = float(input("Enter the value of r (enter -1 to exit): "))
    if r == -1:
        break

    # Initialize world and V_0
    example_world = [[r, -1, 10], [-1, -1, -1], [-1, -1, -1]]
    V_0 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # Call the value_iteration function
    V_k, pi_k, k = value_iteration(example_world, V_0)

    # Print V_k
    print("V_k =")
    print(np.array(V_k))

    # Print pi_k
    print("pi_k =")
    print(np.array(pi_k))


    # print k
    print(f"Converged after k = {k} iterations")
    print("---------------------------------------------------------------")
