from typing import List, Tuple


def check_policy_convergence(current_policy: List[List[str]],
                             last_policy: List[List[str]]) -> bool:
    """
    Function checks for the policy iteration convergence condition, which is
    policy convergence, that is, the last 2 policies are to be the same.
    :param current_policy:
    :param last_policy:
    :return: bool
    """

    for i in range(max(len(current_policy), len(last_policy))):
        for j in range(max(len(current_policy[i]), len(last_policy[i]))):
            try:
                assert current_policy[i][j] == last_policy[i][j]
            except (AssertionError, IndexError):
                return False

    return True


def check_value_convergence(current_values: List[List[float]],
                            last_values: List[List[float]],
                            epsilon: float = 1e-7):
    """

    :param current_values:
    :param last_values:
    :param epsilon:
    :return:
    """
    if len(current_values) == 0 or len(last_values) == 0:
        return False

    for i in range(max(len(current_values), len(last_values))):
        for j in range(max(len(current_values[0]), len(last_values[0]))):
            try:
                assert abs(current_values[i][j] - last_values[i][j]) < epsilon
            except (AssertionError, IndexError):
                return False

    return True


def get_possible_moves(current_position: Tuple[int, int],
                       action: str,
                       board_size: Tuple[int, int]) -> List[Tuple[int, Tuple[int, int]]]:
    """

    :param current_position:
    :param action:
    :param board_size:
    :return:
    """

    possible_moves = []
    moves_directions = None

    if action == 'N':
        moves_directions = [((0, 1), 0.8), ((-1, 0), 0.1), ((1, 0), 0.1)]
    elif action == 'E':
        moves_directions = [((1, 0), 0.8), ((0, 1), 0.1), ((0, -1), 0.1)]
    elif action == 'W':
        moves_directions = [((-1, 0), 0.8), ((0, 1), 0.1), ((0, -1), 0.1)]
    elif action == 'S':
        moves_directions = [((0, -1), 0.8), ((-1, 0), 0.1), ((1, 0), 0.1)]

    for move_dir, prob in moves_directions:
        new_position = tuple(current + offset for current, offset in zip(current_position, move_dir))
        if 0 <= new_position[0] < board_size[0] and 0 <= new_position[1] < board_size[1]:
            possible_moves.append((new_position, prob))
        else:
            possible_moves.append((current_position, prob))

    return possible_moves


def compute_new_value(old_values: List[List[float]],
                      rewards: List[List[float]],
                      policy: List[List[str]],
                      gamma: float):
    """

    :param old_values:
    :param rewards:
    :param policy:
    :param gamma:
    :return:
    """

    size = (len(rewards), len(rewards[0]))
    new_values: List[List[float]] = [[0 for _ in range(size[1])] for _ in range(size[0])]

    for i in range(len(rewards)):
        for j in range(len(rewards[0])):
            if is_terminal(rewards, i, j):
                new_values[i][j] = rewards[i][j]

            else:
                for (s_dash, transition) in get_possible_moves((i, j), policy[i][j], size):
                    new_values[i][j] += transition * (rewards[i][j] + gamma * old_values[s_dash[0]][s_dash[1]])

    return new_values


def is_terminal(rewards: List[List[int]], i: int, j: int):
    if rewards[i][j] != -1:
        return True

    else:
        return False
