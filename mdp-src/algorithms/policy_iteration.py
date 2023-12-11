from utilities.utilities import *


def evaluate_policy(policy: List[List[str]],
                    rewards: List[List[int]],
                    gamma: float) -> List[List[float]]:
    """

    :param policy:
    :param rewards:
    :param gamma:
    :return:
    """

    last_value = []
    current_value = [[0 for _ in range(len(rewards[0]))] for _ in range(len(rewards))]

    while not check_value_convergence(current_value, last_value):
        last_value = current_value

        # update current_value
        current_value = compute_new_value(current_value, rewards, policy, gamma)

    return current_value


def extract_policy(rewards: List[List[int]],
                   policy_values: List[List[float]],
                   gamma: float) -> List[List[str]]:

    new_policy = [["" for _ in range(len(rewards[0]))] for _ in range(len(rewards))]

    for i in range(len(rewards)):
        for j in range(len(rewards[0])):
            max_action_value = - float('inf')
            max_action = ''

            if is_terminal(rewards, i, j):
                new_policy[i][j] = 'T'

            else:

                for action in ['N', 'S', 'E', 'W']:
                    possible_moves = get_possible_moves((i, j), action, (len(rewards), len(rewards[0])))
                    current_action_value = 0
                    for (s_dash, transition) in possible_moves:
                        current_action_value += transition * (rewards[i][j] + gamma * policy_values[s_dash[0]][s_dash[1]])

                    if current_action_value > max_action_value:
                        max_action_value = current_action_value
                        max_action = action

                new_policy[i][j] = max_action

    return new_policy


def policy_iteration(rewards: List[List[int]],
                     initial_policy: List[List[str]],
                     gamma: float,
                     epsilon: float = 1e-7) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Function implements the policy iteration algorithm, using the methods evaluate_policy and extract_policy.

    :param rewards: Represents an array of rewards of the game board.
    :param initial_policy:
    :param gamma: Discount factor
    :param epsilon:
    :return:
    """

    last_policy: List[List[str]] = [[]]
    current_policy: List[List[str]] = initial_policy

    while not check_policy_convergence(current_policy, last_policy):
        last_policy = current_policy

        # Step 1 - Policy Evaluation
        policy_values = evaluate_policy(current_policy, rewards, gamma)

        # Step 2 - Policy Extraction
        current_policy = extract_policy(rewards, policy_values, gamma)

    return current_policy, policy_values


rewards_ = [[100, -1, 10], [-1, -1, -1], [-1, -1, -1]]
initial_policy_ = [['N', 'N', 'N'], ['N', 'N', 'N'], ['N', 'N', 'N']]

# Terminal state condition must be changed

# rewards_ = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]]
# initial_policy_ = [['N', 'N', 'N', 'N'], ['N', 'N', 'N', 'N'], ['N', 'N', 'N', 'N']]

(current_policy, policy_values) = policy_iteration(rewards_, initial_policy_, 0.9)

print(current_policy)

print(policy_values)
